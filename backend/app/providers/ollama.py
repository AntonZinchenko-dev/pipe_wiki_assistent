"""Провайдер локальных моделей через Ollama.

Локальный контур выбран не потому, что модель лучше, а потому, что данные не
должны покидать периметр — ровно та же причина, по которой в нашей вики
SURVEYD не выставлен наружу. Это решение раздела 0: контур определяется
регуляторикой, а не бенчмарком.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import httpx

from ..config import Settings
from .base import (
    ChatChunk, ChatRequest, FinishReason, ProviderError, ProviderUnavailable, ToolCall, Usage,
)
from .resilience import RETRYABLE_STATUS, CircuitBreaker, RetryPolicy, parse_retry_after, with_retry

_FINISH_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "load": FinishReason.ERROR,
}


class OllamaProvider:
    name = "ollama"

    @property
    def model(self) -> str:
        """Модель, которой этот провайдер отвечает.

        Нужно затем, чтобы прогон измерений записывал в шапку ту модель,
        которая ОТВЕЧАЛА, а не ту, что прописана в настройках локального
        провайдера. С двумя провайдерами это перестало быть одним и тем же.
        """
        return self._s.ollama_chat_model

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._s = settings
        # Клиент один на приложение и приходит извне. Новый клиент на запрос —
        # это новое TLS-рукопожатие каждый раз и исчерпание портов под
        # нагрузкой (раздел 2).
        self._client = client
        # ПРЕДОХРАНИТЕЛЬ НА КАЖДУЮ МОДЕЛЬ, а не один на провайдера.
        #
        # Один предохранитель на весь Ollama означал, что отказы РАЗНЫХ моделей
        # складываются в один счётчик. На машине с одной видеокартой это не
        # теория: модель ответов — 7B, модель эмбеддингов — другая, и падают
        # они по разным причинам (первая от нехватки памяти, вторая — почти
        # никогда, она маленькая).
        #
        # Последствие несимметричное и злое. Четыре отказа тяжёлой модели
        # ответов открывали предохранитель и для эмбеддингов — то есть поиск
        # перестал бы работать из-за проблем генерации. А поиск нам нужен
        # именно тогда: без него мы не можем даже честно сказать «нашлось
        # столько-то фрагментов, но ответить не смогли». Система теряла
        # способность объяснить свой отказ по причине, к поиску не
        # относящейся.
        #
        # Аналогия: один автомат защиты на всю квартиру. Замкнуло в стиральной
        # машине — погас и свет, и теперь ты ищешь причину в темноте.
        self._breakers: dict[str, CircuitBreaker] = {}
        self._retry = RetryPolicy()
        self.embed_model = settings.ollama_embed_model
        self.embed_dim = settings.ollama_embed_dim

    def _breaker_for(self, model: str) -> CircuitBreaker:
        breaker = self._breakers.get(model)
        if breaker is None:
            breaker = CircuitBreaker(name=f"{self.name}:{model}")
            self._breakers[model] = breaker
        return breaker

    # ------------------------------------------------------------------ chat

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        # Модель запроса важнее настроенной: человек мог выбрать другую в
        # интерфейсе. Предохранитель берём ИМЕННО для неё — он и заведён на
        # каждую модель отдельно, и брать его по настроенной значило бы
        # считать отказы выбранной модели в чужой счётчик.
        model = request.model or self._s.ollama_chat_model
        breaker = self._breaker_for(model)

        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
                # История идёт ПОСЛЕ вопроса: в ней лежит то, что модель уже
                # просила, и то, что мы ей на это ответили. Без истории
                # второй шаг агента не знает о первом и просит то же самое
                # по кругу, пока не упрётся в ограничение шагов.
                *request.history,
            ],
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.json_schema:
            payload["format"] = request.json_schema
        if request.tools:
            # Пустой список инструментов НЕ отправляем: модель, которой
            # показали пустой набор, начинает извиняться, что ничего не может
            # вызвать, вместо того чтобы просто ответить.
            payload["tools"] = request.tools
        if request.stop:
            payload["options"]["stop"] = request.stop
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        if request.keep_alive is not None:
            payload["keep_alive"] = request.keep_alive
        if request.think is not None:
            payload["think"] = request.think

        # ПОВТОР ДО ПЕРВОГО ОТДАННОГО КУСКА.
        #
        # Раньше потоковый вызов не повторялся вообще, и обоснование звучало
        # разумно: поток, который начал отдавать текст, повторять нельзя —
        # человек увидит начало ответа дважды. Верно. Но из этого не следует,
        # что нельзя повторять поток, который НИЧЕГО не отдал.
        #
        # А именно такие отказы у нас самые частые: Ollama на одной видеокарте
        # отвечает 503, пока выгружает предыдущую модель и загружает нужную.
        # Это временная занятость длиной в секунды. Непотоковый путь
        # (эмбеддинги) её переживал — у него есть with_retry. Потоковый отдавал
        # пользователю «провайдер недоступен» на совершенно исправной системе.
        #
        # Граница безопасности здесь — НЕ «не было текста», а «не было отдано
        # ни одного куска». Куски бывают без текста (служебные, с мыслями
        # модели), и если такой уже ушёл наверх, повтор продублирует его в
        # разборе. Условие поэтому строже, чем кажется нужным.
        #
        # Аналогия: переспросить собеседника можно, пока он не начал отвечать.
        # Как только он сказал первое слово — переспрашивать поздно, надо
        # слушать до конца или честно признать, что связь оборвалась.
        for attempt in range(1, self._retry.attempts + 1):
            breaker.ensure_closed()
            yielded = 0
            emitted_text = False
            failure: ProviderError | None = None

            try:
                async with self._client.stream(
                    "POST", f"{self._s.ollama_url}/api/chat", json=payload,
                    timeout=httpx.Timeout(
                        self._s.request_timeout_s,
                        connect=self._s.connect_timeout_s,
                        # Для потока таймаут чтения — это максимальная ПАУЗА
                        # между кусками, а не общее время ответа. Иначе
                        # законный длинный стрим будет ложно обрываться
                        # (раздел 2). Первый токен ждём дольше: это загрузка
                        # модели, а не молчание. Дальше внутри потока действует
                        # пауза между кусками — см. _iter_ndjson.
                        read=self._s.first_token_timeout_s,
                    ),
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")[:400]
                        raise ProviderError(
                            f"ollama {response.status_code}: {body}",
                            retryable=response.status_code in RETRYABLE_STATUS,
                            status=response.status_code,
                            retry_after_s=parse_retry_after(
                                response.headers.get("retry-after")
                            ),
                            provider=self.name,
                        )

                    async for chunk in self._iter_ndjson(response):
                        if chunk.text:
                            emitted_text = True
                        yielded += 1
                        yield chunk

            except ProviderError as error:
                # Ошибку СТАТУСА мы поднимаем сами, и она не является
                # исключением httpx — значит обработчики ниже её не видят. До
                # этой правки предохранитель на потоковом пути узнавал об
                # отказах ТОЛЬКО из сетевых исключений: сколько бы Ollama ни
                # отвечала 503, он оставался замкнутым, и каждый следующий
                # пользователь платил полным путём — коннект, отправка
                # контекста, ожидание — чтобы получить ту же ошибку.
                failure = error
            except httpx.TimeoutException as error:
                failure = ProviderError(
                    f"ollama замолчал дольше {self._s.stream_idle_timeout_s} с: {error}",
                    # Если текст уже пошёл пользователю — повторять нельзя ни в
                    # каком случае: он увидит начало ответа дважды.
                    retryable=not emitted_text,
                    provider=self.name,
                )
                failure.__cause__ = error
            except httpx.HTTPError as error:
                failure = ProviderUnavailable(
                    f"ollama недоступен ({error.__class__.__name__}): {error}",
                    provider=self.name,
                )
                failure.__cause__ = error

            if failure is None:
                breaker.record_success()
                return

            # Отказ записываем только для временных кодов. 400 и 401 — это наш
            # собственный неверный запрос, и открывать на нём предохранитель
            # значит объявлять провайдера недоступным за свою же ошибку.
            if failure.retryable:
                breaker.record_failure()

            # Модель не понимает переключатель размышлений — снимаем его и
            # повторяем. Один раз, потому что второй уже ничего не изменит.
            #
            # Живёт это ЗДЕСЬ, в адаптере, а не у каждого вызывающего.
            # Раньше такой обход был только у судьи, и в этом была ошибка:
            # «умеет ли модель think» — свойство провайдера и модели, а не
            # свойство того, кто их зовёт. Каждый новый вызывающий
            # наступал бы на те же грабли заново, и наступил: основной
            # ответ на модели без поддержки переключателя падал бы 400 при
            # исправной модели.
            if (
                "think" in payload
                and failure.status == 400
                and "think" in str(failure).lower()
            ):
                payload.pop("think")
                continue

            can_retry = (
                failure.retryable and yielded == 0 and attempt < self._retry.attempts
            )
            if not can_retry:
                raise failure
            await asyncio.sleep(self._retry.delay_for(attempt, failure.retry_after_s))

    async def _iter_ndjson(self, response: httpx.Response) -> AsyncIterator[ChatChunk]:
        """Разбор потока Ollama (NDJSON) через буфер.

        Границы сетевых кусков не совпадают с границами строк JSON: один объект
        спокойно приезжает разрезанным пополам. Разбирать каждый сетевой кусок
        отдельно — это баг, который работает локально и разваливается в проде
        (раздел 5). Поэтому копим в буфере и достаём только законченные строки.
        """
        buffer = ""
        # Два РАЗНЫХ ограничения на один поток, и разделить их приходится
        # самим: httpx умеет только один таймаут чтения на соединение.
        #
        # Ожидание первого токена — это загрузка модели, десятки секунд.
        # Пауза внутри потока — признак того, что провайдер замолчал, и она
        # должна быть короткой. Одно число на оба случая означает выбор между
        # «ложно обрывать холодный старт» и «не замечать зависший поток».
        #
        # Поэтому соединению даём длинный таймаут чтения, а короткую паузу
        # между кусками караулим своим сторожем.
        iterator = response.aiter_text().__aiter__()
        first = True

        while True:
            limit = (
                self._s.first_token_timeout_s if first else self._s.stream_idle_timeout_s
            )
            try:
                raw = await asyncio.wait_for(iterator.__anext__(), timeout=limit)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as error:
                raise httpx.ReadTimeout(
                    f"нет данных дольше {limit:.0f} с "
                    f"({'ожидание первого токена' if first else 'пауза внутри потока'})"
                ) from error
            first = False

            buffer += raw
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                chunk = self._parse_line(line)
                if chunk is not None:
                    yield chunk

        tail = buffer.strip()
        if tail:
            chunk = self._parse_line(tail)
            if chunk is not None:
                yield chunk

    def _parse_line(self, line: str) -> ChatChunk | None:
        """Одна строка потока. Битую строку пропускаем, а не роняем весь поток:
        одна аномалия в середине не должна стоить пользователю всего уже
        сгенерированного текста (раздел 5)."""
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        if data.get("done"):
            reason = _FINISH_MAP.get(data.get("done_reason", "stop"), FinishReason.STOP)
            return ChatChunk(
                done=True,
                finish_reason=reason,
                usage=Usage(
                    prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
                    completion_tokens=int(data.get("eval_count", 0) or 0),
                ),
            )

        message = data.get("message") or {}
        text = message.get("content") or ""
        # Размышления читаем тоже — иначе они исчезают бесследно, и пустой
        # ответ невозможно объяснить.
        thinking = message.get("thinking") or ""
        calls = _parse_tool_calls(message.get("tool_calls"))
        if not text and not thinking and not calls:
            return None
        return ChatChunk(text=text, thinking=thinking, tool_calls=calls)

    # ----------------------------------------------------------------- embed

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async def call() -> list[list[float]]:
            try:
                response = await self._client.post(
                    f"{self._s.ollama_url}/api/embed",
                    json={"model": self.embed_model, "input": texts},
                    timeout=httpx.Timeout(
                        self._s.request_timeout_s, connect=self._s.connect_timeout_s
                    ),
                )
            except httpx.HTTPError as error:
                raise ProviderUnavailable(
                    f"ollama недоступен при эмбеддингах: {error}", provider=self.name
                ) from error

            if response.status_code >= 400:
                raise ProviderError(
                    f"ollama embed {response.status_code}: {response.text[:300]}",
                    retryable=response.status_code in RETRYABLE_STATUS,
                    status=response.status_code,
                    retry_after_s=parse_retry_after(response.headers.get("retry-after")),
                    provider=self.name,
                )

            vectors = response.json().get("embeddings") or []
            if len(vectors) != len(texts):
                raise ProviderError(
                    f"ollama вернул {len(vectors)} векторов на {len(texts)} текстов",
                    retryable=False, provider=self.name,
                )
            if vectors and len(vectors[0]) != self.embed_dim:
                # Размерность не совпала с настроенной — это не «немного не
                # так», это другая модель. Дальше пускать нельзя: в индексе
                # окажется мешанина из несравнимых векторов (раздел 3).
                raise ProviderError(
                    f"размерность вектора {len(vectors[0])} вместо {self.embed_dim}: "
                    f"модель {self.embed_model} не та, что настроена",
                    retryable=False, provider=self.name,
                )
            return vectors

        return await with_retry(
            call, policy=self._retry, breaker=self._breaker_for(self.embed_model)
        )

    # ---------------------------------------------------------- выгрузка

    async def release(self, model: str | None = None) -> bool:
        """Просит Ollama выгрузить модель из памяти немедленно.

        Зачем это нужно. Ollama по умолчанию держит модель загруженной ещё
        пять минут после последнего запроса. На машине с одной видеокартой это
        означает, что попытка обратиться ко второй модели (например к
        судье-модели после отвечающей) приводит не к переключению, а к попытке
        разместить обе — и llama-server падает с ошибкой CUDA.

        Явная выгрузка между этапами дешевле и надёжнее, чем надеяться, что
        провайдер сам догадается освободить память.
        """
        try:
            response = await self._client.post(
                f"{self._s.ollama_url}/api/generate",
                json={"model": model or self._s.ollama_chat_model, "keep_alive": 0},
                timeout=httpx.Timeout(30.0, connect=self._s.connect_timeout_s),
            )
            return response.status_code < 400
        except httpx.HTTPError:
            # Не смогли выгрузить — не повод останавливать работу: это
            # оптимизация памяти, а не обязательный шаг.
            return False

    # ----------------------------------------------------------- список моделей

    async def list_models(self) -> list[str]:
        """Что реально установлено в Ollama.

        Модель эмбеддингов из списка убираем. Она установлена рядом с
        остальными и внешне от них не отличается, но отвечать ею нельзя: это
        другой тип модели, и выбор её в интерфейсе закончился бы невнятной
        ошибкой провайдера вместо честного «так нельзя».

        Проверять больше, чем мы знаем наверняка, не пытаемся: своя модель
        эмбеддингов нам известна из настроек, а угадывать чужие по имени —
        значит однажды спрятать от человека рабочую модель.
        """
        try:
            response = await self._client.get(
                f"{self._s.ollama_url}/api/tags", timeout=httpx.Timeout(5.0, connect=2.0)
            )
            if response.status_code >= 400:
                return []
            names = [str(m.get("model")) for m in response.json().get("models", []) if m.get("model")]
        except (httpx.HTTPError, ValueError):
            return []
        return [name for name in names if name != self.embed_model]

    # ---------------------------------------------------------------- health

    async def health(self) -> dict:
        try:
            response = await self._client.get(
                f"{self._s.ollama_url}/api/tags", timeout=httpx.Timeout(5.0, connect=2.0)
            )
            models = [m.get("model") for m in response.json().get("models", [])]
        except Exception as error:  # health не должен ронять приложение
            return {"provider": self.name, "ok": False, "error": str(error)[:200]}

        return {
            "provider": self.name,
            "ok": True,
            # Предохранителей теперь столько, сколько моделей мы трогали.
            # Показываем ВСЕ: «ollama ok» при открытом предохранителе на
            # модели ответов — это ложь, из-за которой дежурный пойдёт искать
            # причину не там.
            "breakers": {
                model: breaker.state.value for model, breaker in self._breakers.items()
            },
            "chat_model": self._s.ollama_chat_model,
            "embed_model": self.embed_model,
            "chat_model_present": self._s.ollama_chat_model in models,
            "embed_model_present": self.embed_model in models,
            "models": models,
        }


def _parse_tool_calls(raw) -> list[ToolCall]:
    """Разбор вызовов инструментов из ответа Ollama.

    Аргументы приходят объектом, но полагаться на это нельзя: часть моделей
    отдаёт их строкой с JSON внутри, и разница видна только на конкретной
    модели. Приводим к одному виду здесь, в адаптере, — вызывающий код про
    эти различия знать не должен, ровно как и про форму схемы ответа.

    Битый вызов пропускаем молча. Уронить весь поток из-за одного
    непонятного элемента — та же ошибка, что и разбор строки без try: одна
    аномалия не должна стоить пользователю всего ответа.
    """
    if not isinstance(raw, list):
        return []

    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls
