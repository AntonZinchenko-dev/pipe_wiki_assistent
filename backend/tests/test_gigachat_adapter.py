"""Проверки облачного провайдера, которые можно сделать без ключа и без сети.

Смысл файла — поймать то, что иначе выяснится только на живом API и будет
выглядеть как «GigaChat плохо отвечает».

Главное здесь — БЛАНК ОТВЕТА. Поле `json_schema` есть в запросе с самого
начала, локальный провайдер его отправляет, а облачный молча игнорировал.
Работало бы это так: провайдер вернул бы свободный текст, разбор не нашёл бы
ни статуса, ни ссылок, метрики просели бы — и вывод был бы сделан про чужую
модель, а не про наш неполный запрос. Ровно та подмена, которую проект ловит
весь целиком.
"""

from __future__ import annotations

import inspect

from app.providers import gigachat as ga


def test_api_address_is_the_current_one() -> None:
    """Адрес API сменился, и старый дал бы отказ на первом же вызове.

    Адаптер писался вслепую с пометкой «протокол не сверял». При включении
    выяснилось, что угадано почти всё — кроме адреса самого API: выдача токена
    осталась на прежнем хосте, а запросы переехали.
    """
    assert ga.API_URL.startswith("https://api.giga.chat/")
    assert "devices.sberbank.ru" not in ga.API_URL
    # А вот выдача токена осталась там же — это РАЗНЫЕ хосты, и меняются они
    # независимо друг от друга.
    assert ga.OAUTH_URL.startswith("https://ngw.devices.sberbank.ru")


def test_schema_is_sent_in_the_shape_sber_expects() -> None:
    """Форма бланка у Сбера своя, и скопировать её у другого облака нельзя.

    У OpenAI схема завёрнута в объект с именем, у Ollama передаётся полем
    `format`, у Сбера лежит прямо в `response_format.schema`. Тест читает
    исходник, потому что проверить это иначе — значит сходить в сеть.
    """
    source = inspect.getsource(ga.GigaChatProvider.chat_stream)
    assert "response_format" in source
    assert '"type": "json_schema"' in source
    assert '"schema": request.json_schema' in source
    assert '"strict": True' in source


def test_schema_is_only_sent_when_asked_for() -> None:
    """Без схемы поле не появляется: свободный текст — законный режим.

    Судья и отвечающий ходят со схемой, а разовые вызовы могут и без неё.
    Отправлять пустой бланк — значит требовать JSON там, где нужен текст.
    """
    source = inspect.getsource(ga.GigaChatProvider.chat_stream)
    assert "if request.json_schema:" in source


def test_temperature_never_goes_to_exact_zero() -> None:
    """Сервис не принимает ровный ноль, а нам нужен воспроизводимый ответ.

    Ноль поднимается до минимального ненулевого значения в адаптере, а не в
    вызывающем коде: тот про причуды конкретного облака знать не должен.
    """
    assert "max(request.temperature, 0.01)" in inspect.getsource(ga.GigaChatProvider.chat_stream)


def test_health_actually_checks_the_connection() -> None:
    """Проверка связи обязана ходить в сеть, а не пересказывать конфиг.

    Прежняя версия возвращала «ок = ключ прописан в файле». Для облака это
    особенно скверно: отказать оно может из-за сети, ключа, сертификата или
    кончившихся денег, и все четыре причины видны только при живом запросе.
    """
    source = inspect.getsource(ga.GigaChatProvider.health)
    assert "_ensure_token" in source
    assert "MODELS_URL" in source
    assert "stage" in source, "при отказе надо говорить, на каком шаге упало"


def test_disabled_provider_says_why() -> None:
    """Выключенный провайдер сообщает причину, а не просто «не ок»."""
    source = inspect.getsource(ga.GigaChatProvider.health)
    assert "PW_GIGACHAT_ENABLED" in source


def test_certificate_is_isolated_to_this_provider() -> None:
    """Корневой сертификат Сбера не должен расширять доверие всей машины.

    Установленный в хранилище операционной системы, корневой сертификат
    делает доверенной любую подпись этого центра — для любого сайта, не только
    для Сбера, и не только для нашего приложения, а ещё для браузера и почты.
    Ради одного провайдера так расширять круг доверия незачем.

    Поэтому сертификат подключается к ОТДЕЛЬНОМУ клиенту этого провайдера.
    Это не нарушение правила «один клиент на приложение»: правило запрещает
    клиент на ЗАПРОС, из-за рукопожатия TLS каждый раз, а здесь клиент один и
    живёт столько же, сколько приложение.
    """
    source = inspect.getsource(ga.GigaChatProvider.__init__)
    assert "gigachat_ca_bundle" in source
    assert "verify=" in source


def test_own_client_gets_closed() -> None:
    """Свой клиент надо закрыть, иначе при остановке течёт пул соединений.

    Общий клиент закрывает приложение, но про этот оно не знает — значит
    провайдер обязан уметь закрываться сам, а реестр — его об этом попросить.
    """
    assert hasattr(ga.GigaChatProvider, "aclose")

    from app.providers import ProviderRegistry

    assert hasattr(ProviderRegistry, "aclose")


def test_without_a_certificate_the_shared_client_is_used() -> None:
    """Нет сертификата — нет и второго клиента.

    Заводить его «на всякий случай» значит держать лишний пул соединений там,
    где системных корневых сертификатов достаточно.
    """
    source = inspect.getsource(ga.GigaChatProvider.__init__)
    assert "self._own_client or client" in source


def test_missing_certificate_says_which_file_and_which_setting() -> None:
    """Ошибка настройки обязана назвать настройку и путь.

    Без своей проверки httpx роняет приложение на старте сообщением
    «FileNotFoundError: [Errno 2] No such file or directory» — без имени
    файла и без намёка, что речь о сертификате. Стек уходит в недра ssl, и
    человек ищет поломку там.

    Путь в сообщении абсолютный намеренно: относительный считается от каталога
    запуска, а не от файла настроек, и это ровно то место, где промахиваются.
    """
    import httpx
    import pytest

    from app.config import Settings

    settings = Settings(
        gigachat_enabled=True,
        gigachat_auth_key="ключ",
        gigachat_ca_bundle="./нет-такого-файла.cer",
    )
    with pytest.raises(RuntimeError) as failure:
        ga.GigaChatProvider(settings, httpx.AsyncClient())

    message = str(failure.value)
    assert "PW_GIGACHAT_CA_BUNDLE" in message
    assert "нет-такого-файла.cer" in message
    assert "каталога запуска" in message


def test_run_records_the_model_that_actually_answered() -> None:
    """Шапка прогона обязана называть того, кто отвечал.

    В `make_config` жёстко стояло `settings.ollama_chat_model` — кто бы ни
    отвечал. Пока провайдер был один, это совпадало. С появлением облачного
    прогон через него сохранился бы с подписью локальной модели, и через месяц
    два прогона выглядели бы как «одна система, разные цифры».

    Тот же дефект, из-за которого в проект добавляли отпечаток кода:
    конфигурация отвечает на вопрос «та же это система или другая», а не на
    вопрос «что написано в настройках».
    """
    import httpx

    from app.config import Settings
    from app.providers import build_registry

    settings = Settings(
        ollama_chat_model="qwen2.5:14b",
        gigachat_enabled=True,
        gigachat_auth_key="ключ",
        gigachat_model="GigaChat-3-Lightning",
        provider_chain="gigachat,ollama",
    )
    registry = build_registry(settings, httpx.AsyncClient())
    assert registry.answering_model == "GigaChat-3-Lightning"

    settings_local = settings.model_copy(update={"provider_chain": "ollama,gigachat"})
    local = build_registry(settings_local, httpx.AsyncClient())
    assert local.answering_model == "qwen2.5:14b"


def test_the_whole_chain_is_recorded_too() -> None:
    """Одной модели мало: важно, был ли за ней резерв.

    Два прогона на одной модели с разным резервом — разные системы по
    устойчивости.
    """
    from eval.runner import RunConfig

    assert "provider_chain" in RunConfig.__dataclass_fields__


# --------------------------------------------------- расход токенов и финал


def _sse(events: list[str]) -> object:
    """Поддельный ответ, отдающий готовые куски SSE.

    Границы кусков нарочно не совпадают с границами событий: именно на этом
    ломаются разборы, которые читают поток построчно.
    """

    class FakeResponse:
        async def aiter_text(self):
            for piece in events:
                yield piece

    return FakeResponse()


async def _drain(response, **kwargs) -> list:
    return [chunk async for chunk in ga._iter_sse(response, **kwargs)]


def test_token_usage_is_read_from_the_stream() -> None:
    """Расход обязан доехать до финала.

    Прежний разбор читал только текст, и в финал уходили нули. Для локальной
    модели ноль честный — она бесплатная. Для облака ноль означал «счётчик
    денег молчит», причём молчал правдоподобно: в журнале стояла аккуратная
    цифра 0, и отличить «не потратили» от «не посчитали» было нельзя.
    """
    import asyncio

    chunks = asyncio.run(
        _drain(
            _sse([
                'data: {"choices":[{"delta":{"content":"при"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"вет"}}]}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":3327,"completion_tokens":78}}\n\n'
                "data: [DONE]\n\n",
            ])
        )
    )

    final = chunks[-1]
    assert final.done is True
    assert final.usage is not None
    assert final.usage.prompt_tokens == 3327
    assert final.usage.completion_tokens == 78
    # Текст при этом не пострадал и пришёл по кускам, как и был.
    assert "".join(chunk.text for chunk in chunks if chunk.text) == "привет"


def test_truncation_is_not_reported_as_a_normal_finish() -> None:
    """Обрыв по лимиту токенов обязан называться обрывом.

    Разбор всегда возвращал `stop`, а проверка схемы помечает обрезанный
    ответ именно по этой причине. То есть пометка не ставилась никогда — и
    обрезанный на полуслове ответ выглядел для человека законченным.
    """
    import asyncio

    from app.providers.base import FinishReason

    chunks = asyncio.run(
        _drain(
            _sse([
                'data: {"choices":[{"delta":{"content":"нача"}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
                "data: [DONE]\n\n",
            ])
        )
    )
    assert chunks[-1].finish_reason is FinishReason.LENGTH


def test_moderation_refusal_is_not_a_normal_finish_either() -> None:
    """Сработавшая модерация — отказ, а не «модель закончила сама»."""
    import asyncio

    from app.providers.base import FinishReason

    chunks = asyncio.run(
        _drain(_sse(['data: {"choices":[{"delta":{},"finish_reason":"blacklist"}]}\n\ndata: [DONE]\n\n']))
    )
    assert chunks[-1].finish_reason is FinishReason.ERROR


def test_stream_without_done_still_produces_a_final_chunk() -> None:
    """Оборванный поток обязан закончиться финалом.

    Иначе вызывающий код не узнает ни причины остановки, ни расхода, и
    останется ждать события, которого уже не будет.
    """
    import asyncio

    chunks = asyncio.run(
        _drain(_sse(['data: {"choices":[{"delta":{"content":"кусок"}}]}\n\n']))
    )
    assert chunks[-1].done is True


def test_cost_is_a_dash_until_the_tariff_is_set() -> None:
    """Цену не выдумываем.

    Тариф зависит от модели и от счёта, а устаревшая цифра в коде хуже
    отсутствующей: отчёт о расходах выглядел бы точным и был бы неверным.
    Без тарифа показываем токены, а сумму — нет.
    """
    import asyncio

    events = [
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1000,"completion_tokens":500}}\n\ndata: [DONE]\n\n'
    ]
    without = asyncio.run(_drain(_sse(events)))[-1]
    assert without.usage.cost_rub == 0.0

    with_tariff = asyncio.run(_drain(_sse(events), rub_per_1k=2.0))[-1]
    assert with_tariff.usage.cost_rub == 3.0


# ------------------------------------------------- инструменты: третья форма


def test_tools_are_sent_in_the_shape_sber_expects() -> None:
    """У Сбера инструменты называются функциями и лежат без обёртки.

    Нейтральная форма в проекте — форма Ollama и OpenAI: список объектов
    `{"type": "function", "function": {...}}`. Сбер ждёт список САМИХ
    функций в поле `functions`, плюс отдельное `function_call: "auto"`,
    без которого он ничего вызывать не станет.

    Отправить сюда нейтральную форму как есть — это молчаливый отказ от
    инструментов: сервис просто ответит текстом, агент не увидит ни одного
    вызова и решит, что модель считает найденное достаточным. Ровно та же
    подмена, что была со схемой ответа: неполный запрос выглядит как
    решение чужой модели.
    """
    source = inspect.getsource(ga.GigaChatProvider.chat_stream)
    assert 'payload["functions"]' in source
    assert 'tool.get("function", tool)' in source
    assert 'payload["function_call"] = "auto"' in source


def test_function_call_is_parsed_into_the_common_type() -> None:
    """Наверх уходит один тип вызова, одинаковый для всех провайдеров."""
    call = ga._parse_function_call({"name": "search_wiki", "arguments": {"query": "кэш"}})
    assert call is not None
    assert call.name == "search_wiki"
    assert call.arguments == {"query": "кэш"}

    # Строкой — тоже: разница между провайдерами не должна уезжать наверх.
    as_string = ga._parse_function_call({"name": "search_wiki", "arguments": '{"query": "кэш"}'})
    assert as_string == call

    assert ga._parse_function_call(None) is None
    assert ga._parse_function_call({"name": ""}) is None


def test_tool_history_is_translated_too() -> None:
    """Историю вызовов переводит адаптер, а не агент.

    У Ollama это роль `tool` и поле `tool_calls`, у Сбера — роль
    `function` с обязательным именем и поле `function_call`. Без перевода
    второй шаг агента уехал бы в облако в чужой форме: сервис либо не
    понял бы историю, либо отверг бы запрос целиком — и выглядело бы это
    как «агент не работает со Сбером».
    """
    translated = ga._history_for_sber([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search_wiki", "arguments": {"query": "к"}}}],
        },
        {"role": "tool", "name": "search_wiki", "content": "нашлось 3"},
    ])

    assert translated[0]["function_call"] == {"name": "search_wiki", "arguments": {"query": "к"}}
    assert "tool_calls" not in translated[0]
    assert translated[1]["role"] == "function"
    # Имя обязательно: без него сервис не знает, ответ какой функции пришёл.
    assert translated[1]["name"] == "search_wiki"


def test_ordinary_messages_pass_through_untouched() -> None:
    """Перевод трогает только то, что относится к инструментам."""
    message = {"role": "user", "content": "обычное сообщение"}
    assert ga._history_for_sber([message]) == [message]


def test_embedding_models_do_not_show_up_as_answerers() -> None:
    """Сервис отдаёт модели эмбеддингов в общем списке — отвечать ими нельзя.

    Выбор `Embeddings` в списке отвечающих моделей заканчивается невнятной
    ошибкой сервиса, и выглядит это как «облако сломалось», а не как
    «выбрана не та модель». Отбор по имени — компромисс, но знание это об
    ОДНОМ сервисе, и живёт оно в адаптере этого сервиса, рядом со знанием
    про форму его схемы ответа.
    """
    answering = ["GigaChat-2", "GigaChat-2-Max", "GigaChat-3-Lightning", "GigaChat-3-Ultra"]
    embedding = ["Embeddings", "Embeddings-2", "EmbeddingsGigaR", "GigaEmbeddings-3B-2025-09"]

    assert not any(ga._is_embedding_model(name) for name in answering)
    assert all(ga._is_embedding_model(name) for name in embedding)


def test_the_stream_retries_before_the_first_chunk() -> None:
    """У облачного провайдера потоковых повторов не было вовсе.

    Предохранитель стоял, повторов не стояло: 503 или 429 ДО первого токена
    ронял запрос целиком, хотя повторить было можно и безопасно. У
    локального провайдера ровно в этом месте повтор давно работал — два
    адаптера одной системы вели себя по-разному на одной беде.
    """
    source = inspect.getsource(ga.GigaChatProvider.chat_stream)

    assert "for attempt in range(1, self._retry.attempts + 1)" in source
    assert "self._retry.delay_for(attempt, failure.retry_after_s)" in source


def test_the_boundary_is_chunks_not_text() -> None:
    """Граница безопасности — «не отдано ни одного куска», а не «не было текста».

    Куски бывают без текста (служебные), и если такой уже ушёл наверх,
    повтор продублирует его в разборе. Условие строже, чем кажется нужным,
    и это не перестраховка.

    Аналогия: переспросить собеседника можно, пока он не начал отвечать.
    Сказал первое слово — поздно.
    """
    source = inspect.getsource(ga.GigaChatProvider.chat_stream)

    assert "yielded == 0" in source
    assert "emitted_text" not in source, "старое, более слабое условие ушло"


def test_the_token_is_refreshed_between_attempts() -> None:
    """Пауза по Retry-After бывает в минуту, а токен живёт тридцать.

    Повтор со старым токеном вернул бы 401 — неповторяемую ошибку, и
    выглядело бы это как «сервис отказал», а не как «мы пришли с
    просроченным пропуском».
    """
    source = inspect.getsource(ga.GigaChatProvider.chat_stream)

    sleep_at = source.index("asyncio.sleep")
    refresh_at = source.index("token = await self._ensure_token()", sleep_at)

    assert refresh_at > sleep_at, "токен обновляется ПОСЛЕ паузы, а не до неё"


def test_our_own_bad_request_does_not_open_the_breaker() -> None:
    """400 и 401 — наша ошибка, а не недоступность провайдера.

    Открывать на них предохранитель значит объявить сервис недоступным за
    собственный неверный запрос.
    """
    source = inspect.getsource(ga.GigaChatProvider.chat_stream)

    assert "if failure.retryable:\n                self._breaker.record_failure()" in source
