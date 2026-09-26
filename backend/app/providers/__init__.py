"""Сборка провайдеров и цепочка деградации.

Цепочка описана явно: основной провайдер -> резервный -> честная ошибка.
Ни один из шагов не «подразумевается сам собой»: если не написать последний,
по умолчанию получится вечный спиннер (раздел 2 чек-листа).

Кэш как третий шаг деградации появится в шаге про устойчивость — под него уже
оставлено место в `stream_chat`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import AsyncIterator

import httpx

from ..config import Settings
from .base import (
    ChatChunk, ChatProvider, ChatRequest, EmbedProvider, FinishReason,
    ProviderError, ProviderUnavailable, Usage,
)
from .gigachat import GigaChatProvider
from .ollama import OllamaProvider

__all__ = [
    "ChatChunk", "ChatRequest", "FinishReason", "ProviderError", "ProviderUnavailable",
    "Usage", "ProviderRegistry", "build_registry",
]


class ProviderRegistry:
    def __init__(self, providers: dict[str, object], chain: list[str]) -> None:
        self._providers = providers
        self._chain = [name for name in chain if name in providers]
        if not self._chain:
            raise RuntimeError(
                f"пустая цепочка провайдеров: в конфиге {chain}, доступны {list(providers)}"
            )

    @property
    def chain(self) -> list[str]:
        return list(self._chain)

    @property
    def answering_model(self) -> str:
        """Модель, которая реально отвечает, — первая в цепочке.

        Зачем отдельно. Прогон измерений записывал в шапку
        `settings.ollama_chat_model` — жёстко, кто бы ни отвечал. Пока
        провайдер был один, это совпадало. С появлением второго прогон через
        облако сохранился бы с подписью локальной модели, и через месяц два
        прогона выглядели бы как «одна система, разные цифры».

        Это тот же дефект, из-за которого в проект добавляли отпечаток кода:
        конфигурация обязана отвечать на вопрос «та же это система или
        другая», а не на вопрос «что написано в настройках».

        Резерв в подпись не идёт: он вступает, только если основной упал, и
        это видно по ошибкам прогона, а не по шапке.
        """
        provider = self._providers[self._chain[0]]
        return getattr(provider, "model", "") or self._chain[0]

    def get(self, name: str):
        return self._providers[name]

    @property
    def embedder(self) -> EmbedProvider:
        """Эмбеддинги берём только у локальной модели.

        Это не лень, а требование раздела 3: индекс помнит, какой моделью он
        построен, и векторы разных моделей несравнимы. Значит переключение
        провайдера генерации не имеет права трогать эмбеддинги — иначе
        деградация на резерв тихо превратит поиск в выдачу случайных
        документов.
        """
        provider = self._providers.get("ollama")
        if provider is None:
            raise RuntimeError("для эмбеддингов нужен провайдер ollama")
        return provider  # type: ignore[return-value]

    async def list_models(self) -> list[dict]:
        """Что реально можно выбрать прямо сейчас — по всем провайдерам.

        Список собирается опросом, а не из настроек. Настройки говорят, что
        МЫ написали; опрос говорит, что ЕСТЬ. Разошлись эти две вещи ровно
        один раз — и отказ «модели GigaChat не существует» выглядел как
        поломка облака.

        Отвечающая модель помечается: без пометки человек в интерфейсе не
        отличит «выбрано» от «первое в списке».
        """
        default = self.answering_model
        result: list[dict] = []
        for name in self._chain:
            provider = self._providers[name]
            lister = getattr(provider, "list_models", None)
            names = await lister() if lister else []
            if not names:
                # Опрос не удался — показываем хотя бы настроенную модель.
                # Пустой список означал бы «выбирать не из чего», а это
                # неправда: провайдер настроен и, возможно, работает.
                configured = getattr(provider, "model", "")
                names = [configured] if configured else []
            for model in names:
                result.append({
                    "provider": name,
                    "model": model,
                    "is_default": name == self._chain[0] and model == default,
                })
        return result

    def knows(self, provider: str, model: str) -> bool:
        """Есть ли такой провайдер в цепочке. Модель проверяется отдельно.

        Нужно затем, что имя модели приходит из браузера и уходит прямиком в
        чужой API. Пускать туда произвольную строку от клиента нельзя даже
        когда кажется, что ничего страшного не случится.
        """
        return provider in self._providers and provider in self._chain and bool(model)

    async def stream_chat(
        self,
        request: ChatRequest,
        *,
        prefer: str = "",
        on_provider=None,
    ) -> AsyncIterator[ChatChunk]:
        """Идёт по цепочке, пока кто-нибудь не начнёт отдавать текст.

        Ключевая тонкость: переключаться на резерв можно только ДО первого
        отданного пользователю куска. Как только текст пошёл — обрыв означает
        обрыв, а не повод начать заново у другого провайдера, иначе человек
        увидит два разных начала одного ответа.

        `prefer` поднимает выбранного провайдера в начало очереди, но НЕ
        отменяет остальных: человек выбрал, кем отвечать, а не согласился
        остаться без ответа, если тот не отвечает. Порядок резервов при этом
        сохраняется прежний.

        `on_provider` вызывается в момент, когда провайдер отдал первый кусок,
        то есть когда стало известно, кто ОТВЕЧАЕТ. Раньше это имя брали из
        настроек, и в журнале у сберовских ответов стояло имя локальной
        модели — два разных прогона выглядели одинаково подписанными.
        """
        errors: list[str] = []
        order = self._chain
        if prefer in self._providers:
            order = [prefer] + [name for name in self._chain if name != prefer]

        for name in order:
            provider: ChatProvider = self._providers[name]  # type: ignore[assignment]

            # Имя модели принадлежит ОДНОМУ провайдеру.
            #
            # Если человек выбрал облачную модель, а облако не ответило, имя
            # этой модели нельзя тащить в локальную: `GigaChat-3-Lightning`
            # для Ollama — просто несуществующая модель, и резерв упал бы с
            # ошибкой «нет такой модели» вместо того, чтобы спокойно ответить
            # своей. Деградация обязана работать, иначе она не деградация.
            attempt = request
            if request.model and prefer and name != prefer:
                attempt = replace(request, model="")

            emitted = False
            announced = False
            try:
                async for chunk in provider.chat_stream(attempt):
                    if not announced:
                        announced = True
                        if on_provider is not None:
                            on_provider(
                                name,
                                attempt.model or getattr(provider, "model", "") or name,
                            )
                    if chunk.text:
                        emitted = True
                    yield chunk
                return
            except (ProviderError, ProviderUnavailable) as error:
                if emitted:
                    # Поток уже шёл: молча подменить провайдера нельзя.
                    # Отдаём финал с честной причиной — фронт покажет
                    # «ответ прерван» и предложит продолжить (раздел 5).
                    yield ChatChunk(done=True, finish_reason=FinishReason.ERROR, usage=Usage())
                    return
                errors.append(f"{name}: {error}")
                continue

        raise ProviderUnavailable("все провайдеры недоступны — " + "; ".join(errors))

    async def health(self) -> list[dict]:
        result = []
        for name in self._chain:
            provider = self._providers[name]
            checker = getattr(provider, "health", None)
            result.append(await checker() if checker else {"provider": name, "ok": True})
        return result

    async def aclose(self) -> None:
        """Закрыть то, что провайдеры завели сами.

        Общий HTTP-клиент закрывает `lifespan`, но провайдер может держать
        свой — например, с отдельным списком доверенных сертификатов. Про такие
        клиенты снаружи никто не знает, поэтому спрашиваем каждого.
        """
        for provider in self._providers.values():
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                await closer()


def build_registry(settings: Settings, client: httpx.AsyncClient) -> ProviderRegistry:
    providers: dict[str, object] = {"ollama": OllamaProvider(settings, client)}
    if settings.gigachat_enabled:
        providers["gigachat"] = GigaChatProvider(settings, client)
    return ProviderRegistry(providers, settings.providers)
