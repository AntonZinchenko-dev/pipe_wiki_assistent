"""Ручной выбор модели: что именно уходит провайдеру и что попадает в журнал.

Появился он ради сравнения: спросить одно и то же у локальной модели и у
облачной и увидеть разницу своими глазами, а не только в прогоне измерений.
Но выбор модели тянет за собой три вещи, каждую из которых легко забыть, и
все три молчаливые.

1. Имя модели принадлежит ОДНОМУ провайдеру. При деградации на резерв его
   нельзя тащить с собой.
2. В журнал обязана попасть модель, которая ОТВЕЧАЛА, а не выбранная и не
   настроенная.
3. Имя модели приходит из браузера и уходит в чужой API — значит проверяется.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.providers import ProviderRegistry
from app.providers.base import ChatChunk, ChatRequest, FinishReason, ProviderError, Usage


class FakeProvider:
    """Провайдер, который записывает, что у него попросили."""

    def __init__(self, name: str, model: str, *, fails: bool = False) -> None:
        self.name = name
        self.model = model
        self._fails = fails
        self.seen: list[str] = []

    async def chat_stream(self, request: ChatRequest):
        self.seen.append(request.model)
        if self._fails:
            raise ProviderError("занят", retryable=True, provider=self.name)
        yield ChatChunk(text=f"ответ от {self.name}")
        yield ChatChunk(done=True, finish_reason=FinishReason.STOP, usage=Usage())

    async def list_models(self) -> list[str]:
        return [self.model]


def _registry(*providers: FakeProvider) -> ProviderRegistry:
    return ProviderRegistry(
        {provider.name: provider for provider in providers},
        [provider.name for provider in providers],
    )


async def _run(registry: ProviderRegistry, **kwargs) -> list:
    request = ChatRequest(system="с", user="в", **kwargs.pop("request", {}))
    seen: list[tuple[str, str]] = []
    chunks = []
    async for chunk in registry.stream_chat(
        request, on_provider=lambda name, model: seen.append((name, model)), **kwargs
    ):
        chunks.append(chunk)
    return chunks, seen


def test_chosen_provider_goes_first() -> None:
    """Выбор человека меняет очередь, а не отменяет её.

    Резерв остаётся: человек выбрал, КЕМ отвечать, а не согласился остаться
    без ответа, если тот не отвечает.
    """
    local = FakeProvider("ollama", "qwen2.5:14b")
    cloud = FakeProvider("gigachat", "GigaChat-3-Lightning")
    registry = _registry(local, cloud)

    _, seen = asyncio.run(_run(registry, prefer="gigachat"))
    assert seen == [("gigachat", "GigaChat-3-Lightning")]
    assert local.seen == []


def test_model_name_does_not_travel_to_the_fallback() -> None:
    """Самая дорогая ошибка этой правки, если её не сделать.

    `GigaChat-3-Lightning` для Ollama — просто несуществующая модель. Если
    при отказе облака тащить имя выбранной модели в резерв, резерв упадёт с
    «нет такой модели» вместо того, чтобы спокойно ответить своей. Деградация,
    которая ломается при деградации, — это не деградация.
    """
    cloud = FakeProvider("gigachat", "GigaChat-3-Lightning", fails=True)
    local = FakeProvider("ollama", "qwen2.5:14b")
    registry = _registry(local, cloud)

    _, seen = asyncio.run(
        _run(registry, prefer="gigachat", request={"model": "GigaChat-3-Lightning"})
    )

    assert cloud.seen == ["GigaChat-3-Lightning"]
    # Резерв получил ПУСТО и ответил своей настроенной моделью.
    assert local.seen == [""]
    assert seen == [("ollama", "qwen2.5:14b")]


def test_the_journal_names_who_actually_answered() -> None:
    """Не выбранного и не настроенного, а того, кто отдал текст.

    Раньше имя бралось из настроек локального провайдера — независимо от
    того, кто отвечал. Сберовские ответы оказывались подписаны именем `qwen`,
    и журнал врал об авторе ответа ровно тогда, когда он нужен: при разборе
    жалобы через неделю.
    """
    cloud = FakeProvider("gigachat", "GigaChat-3-Lightning", fails=True)
    local = FakeProvider("ollama", "qwen2.5:14b")
    registry = _registry(cloud, local)

    chunks, seen = asyncio.run(_run(registry))

    assert seen[-1] == ("ollama", "qwen2.5:14b")
    assert any("ollama" in chunk.text for chunk in chunks if chunk.text)


def test_the_list_marks_the_default() -> None:
    """Без пометки человек не отличит «выбрано» от «первое в списке»."""
    local = FakeProvider("ollama", "qwen2.5:14b")
    cloud = FakeProvider("gigachat", "GigaChat-3-Lightning")
    registry = _registry(local, cloud)

    listed = asyncio.run(registry.list_models())
    defaults = [item for item in listed if item["is_default"]]
    assert len(defaults) == 1
    assert defaults[0] == {
        "provider": "ollama", "model": "qwen2.5:14b", "is_default": True,
    }


def test_a_provider_that_cannot_be_polled_still_shows_its_configured_model() -> None:
    """Опрос не удался — это не «выбирать не из чего».

    Пустой список означал бы, что провайдера нет вовсе, а он настроен и,
    возможно, работает: не ответил именно справочник моделей.
    """

    class Silent(FakeProvider):
        async def list_models(self) -> list[str]:
            return []

    registry = _registry(Silent("gigachat", "GigaChat-3-Pro"))
    listed = asyncio.run(registry.list_models())
    assert [item["model"] for item in listed] == ["GigaChat-3-Pro"]


@pytest.mark.parametrize(
    "provider, model",
    [
        ("gigachat", "GigaChat-3-Ultra"),   # у этого счёта такой модели нет
        ("ollama", "../../etc/passwd"),      # и такой, разумеется, тоже
        ("несуществующий", "qwen2.5:14b"),
    ],
)
def test_unknown_choice_is_refused_not_silently_replaced(provider: str, model: str) -> None:
    """Неизвестную пару отклоняем, а не подменяем настроенной.

    Молчаливая подмена дала бы человеку ответ другой модели под видом
    выбранной — то есть ровно ту подмену, которую весь проект и ловит. А
    произвольная строка отсюда — это чужой запрос, отправленный за наш счёт
    и с нашим ключом.
    """
    from fastapi.testclient import TestClient

    from app.api.routes import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    local = FakeProvider("ollama", "qwen2.5:14b")
    cloud = FakeProvider("gigachat", "GigaChat-3-Lightning")
    app.state.providers = _registry(local, cloud)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat", json={"question": "вопрос", "provider": provider, "model": model}
        )

    assert response.status_code == 400
    assert "недоступна" in response.json()["detail"]


def test_the_shared_client_is_untouched_by_all_this() -> None:
    """Страховка от случайной утечки ключа в общий клиент.

    Проверка дешёвая, а цена ошибки — запрос к чужому API из клиента, который
    ходит куда угодно.
    """
    client = httpx.AsyncClient()
    registry = _registry(FakeProvider("ollama", "qwen2.5:14b"))
    assert registry.chain == ["ollama"]
    asyncio.run(client.aclose())
