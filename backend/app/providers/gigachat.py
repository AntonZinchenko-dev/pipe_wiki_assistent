"""Провайдер GigaChat — облачный резерв.

Состояние: протокол сверён с документацией при включении провайдера.

Адаптер писался вслепую, с честной пометкой «протокол не проверял». При
сверке выяснилось, что угадано было почти всё: адрес выдачи токена, Basic-схема
авторизации, заголовок RqUID, формат срока жизни токена в миллисекундах. Не
угадано два места, и оба существенные.

ПЕРВОЕ: адрес запросов к самому API сменился. Было
`gigachat.devices.sberbank.ru/api/v1`, стало `api.giga.chat/v1`. Старый адрес
дал бы отказ на первом же вызове — это как раз тот случай, когда «форма
правильная» не спасает.

ВТОРОЕ, и оно важнее: СХЕМА ОТВЕТА НЕ ПЕРЕДАВАЛАСЬ ВООБЩЕ. Поле
`request.json_schema` есть в запросе с самого начала, локальный провайдер его
отправляет, а этот молча игнорировал. Работать оно бы «работало»: провайдер
вернул бы свободный текст, разбор ответа не нашёл бы ни статуса, ни ссылок, и
выглядело бы это как «GigaChat плохо отвечает», а не как «мы забыли попросить
бланк». Ровно та подмена, которую мы весь проект и ловим: неполный вызов
выглядит как плохое качество чужой системы.

У Сбера бланк задаётся полем `response_format` и устроен иначе, чем у OpenAI:
схема лежит прямо в `response_format.schema`, а не в отдельном объекте с
именем. Поэтому просто скопировать вызов от другого облака нельзя.

Что здесь важно по чек-листу и не зависит от версии протокола:

- токен обновляется ЗАРАНЕЕ, по времени истечения, а не по факту 401. Схема
  «поймали 401 — обновили — повторили» рвёт активный стрим: часть ответа уже
  ушла человеку (раздел 2);
- ключ живёт в переменных окружения на сервере и никогда не уходит в браузер;
- ошибки размечаются на retryable и нет — тем же способом, что у локального
  провайдера, потому что вызывающий код не должен знать разницы.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import httpx

from ..config import Settings
from .base import ChatChunk, ChatRequest, ProviderError, ProviderUnavailable, ToolCall
from .resilience import RETRYABLE_STATUS, CircuitBreaker, RetryPolicy, parse_retry_after

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
# Адрес API, а не выдачи токена: это разные хосты, и меняются они независимо.
# Прежний `gigachat.devices.sberbank.ru/api/v1` в документации больше не
# значится.
API_URL = "https://api.giga.chat/v1/chat/completions"
MODELS_URL = "https://api.giga.chat/v1/models"
# Токен живёт полчаса. Обновляем за пять минут до конца, а не по отказу 401:
# схема «поймали 401 — обновили — повторили» рвёт начатый поток.
TOKEN_SAFETY_WINDOW_S = 300.0


class GigaChatProvider:
    name = "gigachat"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._s = settings
        self._breaker = CircuitBreaker(name=self.name)
        self._retry = RetryPolicy()
        self._token: str = ""
        self._token_expires_at: float = 0.0

        # СВОЙ HTTP-клиент, если задан корневой сертификат Сбера.
        #
        # Это не нарушение правила «один клиент на приложение». Правило
        # запрещает клиент НА ЗАПРОС — из-за нового рукопожатия TLS каждый раз
        # и исчерпания портов под нагрузкой. Здесь клиент один на провайдера и
        # живёт столько же, сколько приложение.
        #
        # А нужен он затем, что список доверенных сертификатов задаётся при
        # создании клиента и на всех его соединениях один. Подмешать корневой
        # сертификат Сбера в общий клиент значило бы доверять этому центру во
        # всех исходящих соединениях приложения — ровно то расширение круга
        # доверия, которого мы избегаем, не ставя сертификат в систему.
        self._own_client: httpx.AsyncClient | None = None
        if settings.gigachat_ca_bundle:
            # Проверяем существование файла САМИ, до создания клиента.
            #
            # Без этого httpx роняет приложение на старте с сообщением
            # «FileNotFoundError: [Errno 2] No such file or directory» — без
            # имени файла и без единого намёка, что речь о сертификате. Стек
            # при этом уходит в недра ssl, и человек ищет поломку там.
            #
            # Правило общее: ошибка конфигурации обязана называть настройку и
            # путь. Путь печатаем абсолютный, потому что относительный
            # считается от каталога запуска, а не от файла настроек, и это
            # ровно то место, где обычно и промахиваются.
            bundle = Path(settings.gigachat_ca_bundle)
            if not bundle.is_file():
                raise RuntimeError(
                    f"не найден корневой сертификат Сбера: {bundle.resolve()}\n"
                    f"  настройка PW_GIGACHAT_CA_BUNDLE = {settings.gigachat_ca_bundle}\n"
                    f"  путь считается от каталога запуска: {Path.cwd()}\n"
                    "  Скачать: https://developers.sber.ru/docs/ru/gigachat/certificates\n"
                    "  Либо уберите настройку — тогда возьмутся системные сертификаты."
                )
            self._own_client = httpx.AsyncClient(
                verify=settings.gigachat_ca_bundle,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                timeout=httpx.Timeout(
                    settings.request_timeout_s, connect=settings.connect_timeout_s
                ),
            )
        self._client = self._own_client or client

    async def aclose(self) -> None:
        """Закрыть свой клиент, если он создавался.

        Без этого при остановке приложения остаётся открытый пул соединений:
        общий клиент закрывает `lifespan`, а про этот он не знает.
        """
        if self._own_client is not None:
            await self._own_client.aclose()
            self._own_client = None

    @property
    def model(self) -> str:
        return self._s.gigachat_model

    @property
    def configured(self) -> bool:
        return bool(self._s.gigachat_enabled and self._s.gigachat_auth_key)

    async def _ensure_token(self) -> str:
        """Токен обновляется до истечения, а не по отказу."""
        if self._token and time.time() < self._token_expires_at - TOKEN_SAFETY_WINDOW_S:
            return self._token

        try:
            response = await self._client.post(
                OAUTH_URL,
                headers={
                    "Authorization": f"Basic {self._s.gigachat_auth_key}",
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"scope": self._s.gigachat_scope},
                timeout=httpx.Timeout(20.0, connect=self._s.connect_timeout_s),
            )
        except httpx.HTTPError as error:
            raise ProviderUnavailable(
                f"gigachat: не удалось получить токен ({error})", provider=self.name
            ) from error

        if response.status_code >= 400:
            raise ProviderError(
                f"gigachat oauth {response.status_code}: {response.text[:200]}",
                retryable=response.status_code in RETRYABLE_STATUS,
                status=response.status_code, provider=self.name,
            )

        payload = response.json()
        self._token = payload["access_token"]
        # Сервис отдаёт время истечения в миллисекундах epoch.
        self._token_expires_at = float(payload.get("expires_at", 0)) / 1000.0
        return self._token

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        if not self.configured:
            raise ProviderUnavailable("gigachat выключен в конфигурации", provider=self.name)
        self._breaker.ensure_closed()
        token = await self._ensure_token()

        payload = {
            "model": request.model or self._s.gigachat_model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
                # История тоже переводится: у Сбера вызов функции лежит в
                # `function_call` у ассистента, а ответ приходит ролью
                # `function` с обязательным именем. У Ollama то же самое
                # называется `tool_calls` и ролью `tool`. Четвёртая форма
                # одного и того же в этом файле.
                *_history_for_sber(request.history),
            ],
            "temperature": max(request.temperature, 0.01),  # сервис не любит ровный ноль
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        # БЛАНК ОТВЕТА. Без него провайдер вернёт свободный текст, разбор не
        # найдёт ни статуса, ни ссылок — и это будет выглядеть как плохое
        # качество модели, а не как наш неполный запрос.
        #
        # Форма у Сбера своя: схема лежит прямо в `response_format.schema`. У
        # OpenAI она завёрнута в объект с именем, у Ollama передаётся полем
        # `format`. Три провайдера — три способа сказать одно и то же, и
        # поэтому перевод формы живёт в адаптере: вызывающий код отдаёт
        # `json_schema` и про эти различия не знает.
        #
        # `strict` включён осознанно. Мягкий режим — это «постарайся», то есть
        # те же несколько процентов брака, ради избавления от которых схема и
        # затевалась.
        if request.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "schema": request.json_schema,
                "strict": True,
            }

        # ИНСТРУМЕНТЫ. Третья форма одного и того же, и снова своя.
        #
        # У Ollama это `tools` со списком объектов `{"type":"function",
        # "function":{...}}`. У Сбера — `functions` со списком САМИХ функций,
        # без обёртки, плюс отдельное поле `function_call: "auto"`, без
        # которого сервис вызывать ничего не станет.
        #
        # Перевод формы живёт здесь по той же причине, что и перевод схемы:
        # вызывающий код описывает инструмент один раз, в нейтральном виде, и
        # про различия провайдеров не знает. Иначе каждый новый провайдер
        # означал бы правку агента.
        if request.tools:
            payload["functions"] = [
                tool.get("function", tool) for tool in request.tools
            ]
            payload["function_call"] = "auto"


        emitted_text = False
        try:
            async with self._client.stream(
                "POST", API_URL, json=payload,
                headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
                timeout=httpx.Timeout(
                    self._s.request_timeout_s,
                    connect=self._s.connect_timeout_s,
                    read=self._s.stream_idle_timeout_s,
                ),
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")[:300]
                    raise ProviderError(
                        f"gigachat {response.status_code}: {body}",
                        retryable=response.status_code in RETRYABLE_STATUS,
                        status=response.status_code,
                        retry_after_s=parse_retry_after(response.headers.get("retry-after")),
                        provider=self.name,
                    )
                async for chunk in _iter_sse(
                    response, rub_per_1k=self._s.gigachat_rub_per_1k
                ):
                    if chunk.text:
                        emitted_text = True
                    yield chunk
        except httpx.HTTPError as error:
            self._breaker.record_failure()
            raise ProviderError(
                f"gigachat: обрыв потока ({error})",
                retryable=not emitted_text, provider=self.name,
            ) from error
        else:
            self._breaker.record_success()

    async def list_models(self) -> list[str]:
        """Какие модели доступны ЭТОМУ счёту.

        Список спрашиваем у сервиса, а не держим в коде. Причина простая:
        зашитый список устаревает молча. Именно на этом мы уже обожглись —
        в настройках стояло имя `GigaChat`, которого у сервиса больше нет, и
        отказ выглядел как «облако не работает», а не как «такой модели нет».

        Ошибку наверх не поднимаем: список моделей нужен для выпадающего
        списка в интерфейсе, и падать целиком из-за него нельзя. Пустой
        список означает «спросить не удалось» — здоровье провайдера про
        причину расскажет отдельно и подробно.
        """
        if not self.configured:
            return []
        try:
            await self._ensure_token()
            response = await self._client.get(
                MODELS_URL,
                headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
                timeout=httpx.Timeout(10.0, connect=self._s.connect_timeout_s),
            )
            if response.status_code >= 400:
                return []
            return [
                name
                for name in (
                    str(item.get("id")) for item in (response.json().get("data") or [])
                    if item.get("id")
                )
                if not _is_embedding_model(name)
            ]
        except (httpx.HTTPError, ProviderError, ProviderUnavailable, ValueError):
            return []

    async def health(self) -> dict:
        """Настоящая проверка связи, а не отчёт о содержимом конфига.

        Прежняя версия возвращала `ok = настроен ли провайдер` — то есть
        отвечала «да» на вопрос «прописан ли ключ в файле», притворяясь ответом
        на вопрос «работает ли». Для облачного провайдера это особенно скверно:
        отказать он может по четырём независимым причинам — нет сети, не
        принят ключ, не установлен корневой сертификат, кончились деньги на
        счету, — и все четыре видны только при настоящем запросе.

        Проверяем двумя шагами, потому что ломаются они по-разному: сначала
        выдача токена, потом список моделей. Если упал первый — дело в ключе
        или сертификате. Если второй — ключ принят, а до API не достучались.
        """
        if not self.configured:
            return {
                "provider": self.name, "ok": False, "configured": False,
                "breaker": self._breaker.state.value,
                "note": "выключен: нет PW_GIGACHAT_ENABLED или PW_GIGACHAT_AUTH_KEY",
            }

        base = {
            "provider": self.name, "configured": True,
            "breaker": self._breaker.state.value, "model": self._s.gigachat_model,
        }
        try:
            await self._ensure_token()
        except (ProviderError, ProviderUnavailable) as error:
            return {**base, "ok": False, "stage": "токен", "error": str(error)[:200]}

        try:
            response = await self._client.get(
                MODELS_URL,
                headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
                timeout=httpx.Timeout(10.0, connect=self._s.connect_timeout_s),
            )
        except httpx.HTTPError as error:
            return {**base, "ok": False, "stage": "список моделей", "error": str(error)[:200]}

        if response.status_code >= 400:
            return {
                **base, "ok": False, "stage": "список моделей",
                "error": f"{response.status_code}: {response.text[:200]}",
            }

        names = [item.get("id") for item in (response.json().get("data") or [])]
        return {**base, "ok": True, "models": names[:10]}


async def _iter_sse(response: httpx.Response, *, rub_per_1k: float = 0.0) -> AsyncIterator[ChatChunk]:
    """Разбор SSE через буфер — та же причина, что и в Ollama: границы сетевых
    кусков не совпадают с границами событий.

    Помимо текста отсюда обязаны выйти ещё две вещи, и раньше обе терялись.

    РАСХОД ТОКЕНОВ. Сбер присылает его в последнем событии перед `[DONE]`.
    Прежний разбор читал только `delta.content`, поэтому в финал уходил
    `Usage()` — нули. Для локальной модели ноль честный, она бесплатная; для
    облака ноль означал «счётчик денег молчит». Хуже того, молчал он
    правдоподобно: в интерфейсе и в журнале стояла аккуратная цифра 0, а не
    прочерк, и отличить «не потратили» от «не посчитали» было нельзя.

    ПРИЧИНА ЗАВЕРШЕНИЯ. Она приходит в `choices[0].finish_reason`. Прежний
    разбор всегда отдавал `STOP` — то есть ответ, обрезанный по лимиту
    токенов, приходил наверх как законченный сам собой. А проверка схемы
    полагается ровно на эту причину, чтобы пометить обрыв; с постоянным `STOP`
    пометка не ставилась никогда.

    Цену считаем здесь же и только если тариф задан. Ноль при незаданном
    тарифе — это по-прежнему «не посчитали», но теперь рядом есть настоящие
    токены, по которым видно, что расход был.
    """
    import json

    from .base import FinishReason, Usage

    finish_map = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        # Модель попросила инструмент. Это законное завершение шага, а не
        # ошибка: агент выполнит вызов и спросит ещё раз.
        "function_call": FinishReason.STOP,
        # Сработала модерация Сбера. Это не «модель закончила сама»: ответа
        # нет, и называть такой финал обычной остановкой значит прятать отказ.
        "blacklist": FinishReason.ERROR,
        "error": FinishReason.ERROR,
    }

    def final(usage_raw: dict | None, reason: str) -> ChatChunk:
        prompt = int((usage_raw or {}).get("prompt_tokens") or 0)
        completion = int((usage_raw or {}).get("completion_tokens") or 0)
        usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_rub=round((prompt + completion) / 1000.0 * rub_per_1k, 4) if rub_per_1k else 0.0,
        )
        return ChatChunk(
            done=True, finish_reason=finish_map.get(reason, FinishReason.STOP), usage=usage
        )

    buffer = ""
    last_usage: dict | None = None
    last_reason = "stop"

    async for raw in response.aiter_text():
        buffer += raw
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    yield final(last_usage, last_reason)
                    return
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue  # битое событие пропускаем, поток живёт
                # Расход приходит отдельным полем последнего события, а не
                # внутри choices, поэтому читаем его до разбора текста.
                if isinstance(parsed.get("usage"), dict):
                    last_usage = parsed["usage"]
                choice = (parsed.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    last_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                call = _parse_function_call(delta.get("function_call"))
                if call is not None:
                    yield ChatChunk(tool_calls=[call])
                text = delta.get("content") or ""
                if text:
                    yield ChatChunk(text=text)

    # Поток кончился без `[DONE]` — финал всё равно обязан быть, иначе
    # вызывающий код не узнает ни причины, ни расхода.
    yield final(last_usage, last_reason)


def _parse_function_call(raw) -> ToolCall | None:
    """Разбор вызова функции из потока Сбера.

    Аргументы приходят объектом, но встречается и строка с JSON внутри —
    как и у локального провайдера. Приводим к одному виду здесь: наверх
    уходит один тип `ToolCall`, одинаковый для всех провайдеров.
    """
    import json

    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    arguments = raw.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(name=name, arguments=arguments)


def _history_for_sber(history: list[dict]) -> list[dict]:
    """Перевод истории инструментов в форму Сбера.

    Нейтральной формой в проекте считается форма Ollama — она же форма
    OpenAI, и вызывающий код пишет только её. Здесь она превращается в то,
    что понимает этот сервис.

    Смысл ровно тот же, что у перевода схемы ответа и списка инструментов:
    знание о различиях провайдеров живёт в адаптере провайдера. Иначе агент
    пришлось бы править при каждом новом сервисе, а он про сервисы не знает
    и знать не должен.
    """
    result: list[dict] = []
    for message in history:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            call = (message["tool_calls"][0] or {}).get("function") or {}
            result.append({
                "role": "assistant",
                "content": "",
                "function_call": {
                    "name": call.get("name", ""),
                    "arguments": call.get("arguments") or {},
                },
            })
        elif role == "tool":
            result.append({
                "role": "function",
                # Имя обязательно: без него сервис не понимает, ответ какой
                # именно функции ему прислали.
                "name": message.get("name", ""),
                "content": message.get("content", ""),
            })
        else:
            result.append(message)
    return result


def _is_embedding_model(name: str) -> bool:
    """Модель эмбеддингов, а не собеседник.

    Сервис отдаёт их в общем списке моделей, вперемешку с отвечающими:
    `Embeddings`, `EmbeddingsGigaR`, `GigaEmbeddings-3B-…`. Выбор такой в
    списке ответов заканчивается невнятной ошибкой сервиса, и выглядит это
    как «облако сломалось», а не как «выбрана не та модель».

    Отбор по имени — компромисс, и он осознанный. В общем случае угадывать
    назначение модели по названию нельзя: однажды спрячешь от человека
    рабочую. Но здесь это знание об ОДНОМ конкретном сервисе, и живёт оно в
    адаптере этого сервиса — там же, где знание про форму его схемы ответа
    и про имя поля с функциями. У локального провайдера та же задача решена
    иначе: там мы точно знаем имя своей модели эмбеддингов из настроек, и
    гадать не нужно.
    """
    return "embed" in name.casefold()
