"""Тесты устойчивости: предохранитель, повторы, ключ лимитера.

Все четыре дефекта здесь были найдены сплошным аудитом и отложены как
«не падает, но неверно». Каждый из них ломает систему не в момент правки, а
в момент аварии — то есть тогда, когда проверить уже нечем.
"""

import pytest

from app.providers.base import ProviderUnavailable
from app.providers.resilience import BreakerState, CircuitBreaker


def test_half_open_admits_exactly_one_probe():
    """Полуоткрытый предохранитель обязан пускать ОДИН запрос.

    Так было написано в комментарии и не было в коде: состояние только
    читалось, и все ожидающие проходили одновременно.

    Как это ломает прод. Провайдер лежит, предохранитель открыт, запросы
    копятся (они быстро отказываются, а люди нажимают снова). Через
    двадцать секунд состояние становится полуоткрытым — и вся очередь уходит
    в провайдера одним залпом, ровно когда он начал вставать. Он падает
    снова, цикл повторяется, и мы так и не узнаём, был ли он готов.
    """
    breaker = CircuitBreaker(failure_threshold=2, recovery_time_s=0.0, name="t")
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is BreakerState.HALF_OPEN

    breaker.ensure_closed()  # пробный проходит
    with pytest.raises(ProviderUnavailable, match="пробный запрос уже выполняется"):
        breaker.ensure_closed()

    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    breaker.ensure_closed()  # замкнут — проходят все


def test_probe_flag_clears_on_failure_so_the_breaker_does_not_jam():
    """Неудачный пробный запрос не должен заклинить предохранитель навсегда.

    Обратная сторона правила «один пробный»: если флаг не снимать, второй
    пробный не пройдёт никогда, и провайдер останется «недоступным» даже
    после того, как встанет. Лечение хуже болезни.
    """
    breaker = CircuitBreaker(failure_threshold=1, recovery_time_s=0.0, name="t")
    breaker.record_failure()
    breaker.ensure_closed()
    breaker.record_failure()          # пробный не удался
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.ensure_closed()           # следующий пробный снова возможен


def test_breaker_is_per_model_not_per_provider():
    """Отказы разных моделей не должны складываться в один счётчик.

    На машине с одной видеокартой модель ответов (7B) падает от нехватки
    памяти, а модель эмбеддингов — почти никогда. Общий предохранитель
    означал, что четыре отказа генерации выключают поиск: система теряла
    способность даже честно сказать «фрагменты нашлись, ответить не смог».

    Аналогия: один автомат защиты на всю квартиру. Замкнуло в стиральной
    машине — погас свет, и причину ты ищешь в темноте.
    """
    import httpx

    from app.config import Settings
    from app.providers.ollama import OllamaProvider

    provider = OllamaProvider(Settings(), httpx.AsyncClient())
    chat = provider._breaker_for("qwen2.5:14b")
    embed = provider._breaker_for("bge-m3:latest")
    assert chat is not embed

    for _ in range(4):
        chat.record_failure()
    assert chat.state is BreakerState.OPEN
    assert embed.state is BreakerState.CLOSED
    embed.ensure_closed()  # поиск продолжает работать


def test_stream_retries_only_before_the_first_chunk():
    """Повторять поток можно, пока наверх не ушёл ни один кусок.

    Раньше потоковый путь не повторялся вообще: обоснование «человек увидит
    начало ответа дважды» верно, но из него не следует, что нельзя повторять
    поток, который НИЧЕГО не отдал. А это самый частый у нас отказ: Ollama на
    одной видеокарте отвечает 503, пока меняет модель в памяти. Непотоковый
    путь это переживал, потоковый отдавал «провайдер недоступен» на исправной
    системе.

    Граница безопасности — не «не было текста», а «не было ни одного куска»:
    служебные куски без текста тоже уходят наверх, и их дубль испортит разбор.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "app" / "providers" / "ollama.py").read_text(
        encoding="utf-8"
    )
    stream = source[source.index("async def chat_stream("):source.index("async def _iter_ndjson(")]

    assert "for attempt in range(1, self._retry.attempts + 1)" in stream
    assert "yielded == 0" in stream, "условие повтора обязано быть про отданные куски"
    assert "emitted_text" in stream, "текст пользователю по-прежнему запрещает повтор"


def test_client_key_is_the_real_client_behind_a_proxy():
    """За прокси лимит обязан считаться на пользователя, а не на всех вместе.

    Адрес соединения за nginx или Cloudflare — это адрес прокси, один для
    всех. Лимитер в такой схеме считает «тридцать запросов в минуту на всю
    компанию»: один активный человек исчерпывает квоту, остальные получают 429
    на пустой системе — и в логах всё выглядит исправно.
    """
    from app.ratelimit import client_key

    headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1, 10.0.0.2"}
    assert client_key(headers, "10.0.0.2", trust_proxy=True) == "203.0.113.7"


def test_client_key_ignores_forwarded_header_when_proxy_is_not_trusted():
    """Без прокси впереди заголовок ставит сам клиент — верить ему нельзя.

    Иначе лимит обходится сменой одной цифры, и лимитер не ослаблен, а
    выключен при полной видимости работы. Поэтому доверие включается явно, а
    по умолчанию выключено: безопасное значение — то, которое не создаёт дыру
    в незнакомой схеме развёртывания.
    """
    from app.ratelimit import client_key

    headers = {"x-forwarded-for": "1.2.3.4"}
    assert client_key(headers, "10.0.0.2", trust_proxy=False) == "10.0.0.2"
    assert client_key({}, None, trust_proxy=True) == "unknown"


def test_debug_search_endpoint_fuses_like_production():
    """Отладочный поиск обязан сливать выдачу так же, как продовый.

    Расхождение здесь превращает инструмент диагностики в источник ложных
    выводов: человек смотрит на один порядок фрагментов, а модель получает
    другой — и разбор поломки уходит не туда.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "app" / "api" / "routes.py").read_text(
        encoding="utf-8"
    )
    assert "keyword_weight=state.settings.keyword_weight" in source
