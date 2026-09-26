"""Повторы и предохранитель.

Три вещи, которые кажутся мелочью и каждая из которых ломает прод по-своему:

1. Повторять только временные ошибки. 400 и 401 повтором не лечатся — они не
   про перегрузку, а про то, что запрос неверен; повтор лишь втрое замедлит
   ответ пользователю.
2. Джиттер в задержке. Без него сотня клиентов, отвалившихся одновременно,
   повторит синхронно и добьёт провайдер как раз в момент восстановления.
3. Предохранитель. Пока провайдер лежит, не надо в него долбить: и ему мешаем
   встать, и пользователю отдаём таймаут вместо быстрого честного отказа.

Раздел 2 чек-листа.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum

from .base import ProviderError, ProviderUnavailable

# Коды, при которых повтор имеет смысл. Список закрытый и явный: «повторяем
# всё, что не 2xx» — это способ незаметно ретраить собственные баги.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class BreakerState(str, Enum):
    CLOSED = "closed"        # работаем как обычно
    OPEN = "open"            # отказываем сразу, не трогая провайдера
    HALF_OPEN = "half_open"  # пробуем один запрос


@dataclass
class CircuitBreaker:
    """Предохранитель на одного провайдера.

    Состояние держим в памяти процесса. Для одного инстанса этого достаточно;
    когда инстансов станет несколько, состояние переедет в Redis — но логика
    останется той же.
    """

    failure_threshold: int = 4
    recovery_time_s: float = 20.0
    name: str = ""

    _state: BreakerState = field(default=BreakerState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _probing: bool = field(default=False, init=False)

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_time_s:
                self._state = BreakerState.HALF_OPEN
        return self._state

    def ensure_closed(self) -> None:
        """Бросает, если провайдера трогать нельзя. Вызывать ДО запроса."""
        state = self.state
        if state is BreakerState.OPEN:
            left = self.recovery_time_s - (time.monotonic() - self._opened_at)
            raise ProviderUnavailable(
                f"предохранитель открыт, следующая попытка через {left:.0f} с",
                provider=self.name,
            )

        if state is BreakerState.HALF_OPEN:
            # ПОЛУОТКРЫТОЕ СОСТОЯНИЕ ПУСКАЕТ РОВНО ОДИН ЗАПРОС.
            #
            # Так было написано в комментарии («пробуем один запрос»), но не в
            # коде: состояние только читалось, и все, кто ждал, проходили
            # одновременно. Это ломает ровно то, ради чего предохранитель
            # ставят.
            #
            # Как это выглядит в проде. Провайдер лежит, предохранитель
            # открыт, запросы копятся — их ведь никуда не отправляют, они
            # быстро отказываются, а пользователи нажимают снова. Через
            # двадцать секунд состояние становится полуоткрытым, и ВСЯ
            # накопившаяся очередь уходит в провайдера одним залпом — ровно в
            # ту секунду, когда он только начал вставать. Он падает снова, и
            # цикл повторяется с тем же исходом.
            #
            # Аналогия: в магазин после аварии пускают «на пробу». Если внутрь
            # заходит один покупатель — видно, работает ли касса. Если вся
            # очередь сразу — касса ляжет опять, и мы так и не узнаем, была ли
            # она готова.
            if self._probing:
                raise ProviderUnavailable(
                    "предохранитель полуоткрыт: пробный запрос уже выполняется",
                    provider=self.name,
                )
            self._probing = True

    def record_success(self) -> None:
        self._failures = 0
        self._probing = False
        self._state = BreakerState.CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        self._probing = False
        # В полуоткрытом состоянии одна неудача возвращает нас в открытое:
        # пробный запрос на то и пробный.
        if self._state is BreakerState.HALF_OPEN or self._failures >= self.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_at = time.monotonic()
            self._failures = 0


@dataclass(slots=True)
class RetryPolicy:
    attempts: int = 3
    base_delay_s: float = 0.6
    max_delay_s: float = 8.0
    jitter: float = 0.3  # доля от задержки, ±

    def delay_for(self, attempt: int, retry_after_s: float | None) -> float:
        """Задержка перед следующей попыткой.

        Если провайдер прислал Retry-After — слушаем его: он знает про свою
        очередь больше, чем наша эвристика. Джиттер добавляем в любом случае,
        чтобы сотня клиентов не пришла в одну и ту же секунду.
        """
        if retry_after_s is not None:
            # Retry-After НЕ обрезается по max_delay_s. Раньше обрезался, и
            # при «Retry-After: 60» мы повторяли через восемь секунд, получали
            # второй отказ, снова через восемь, попытки кончались. Мы и не
            # подождали как просили, и добавили провайдеру две лишние отбитые
            # попытки ровно в момент перегрузки. Предел задержки — защита от
            # нашего собственного расчёта, а не от прямого указания сервера.
            base = retry_after_s
        else:
            base = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        spread = base * self.jitter
        if retry_after_s is not None:
            # Джиттер к Retry-After добавляется ТОЛЬКО В ПЛЮС. Двусторонний
            # разброс позволял прийти раньше, чем сервер разрешил: при
            # «Retry-After: 2» мы могли повторить через 1.4 секунды — то есть
            # нарушить прямое указание, пытаясь его соблюсти.
            return base + random.uniform(0.0, spread)
        return max(0.0, base + random.uniform(-spread, spread))


async def with_retry(operation, *, policy: RetryPolicy, breaker: CircuitBreaker | None = None):
    """Выполняет НЕпотоковую операцию с повторами.

    Важно: для потоковых вызовов эта обёртка не годится. Поток, который уже
    начал отдавать текст, повторять нельзя — решение о повторе там принимается
    внутри самого потока, до первого отданного куска (см. providers/ollama).
    """
    last: Exception | None = None
    for attempt in range(1, policy.attempts + 1):
        if breaker:
            breaker.ensure_closed()
        try:
            result = await operation()
        except ProviderError as error:
            last = error
            if breaker:
                breaker.record_failure()
            if not error.retryable or attempt == policy.attempts:
                raise
            await asyncio.sleep(policy.delay_for(attempt, error.retry_after_s))
        else:
            if breaker:
                breaker.record_success()
            return result
    raise last if last else RuntimeError("with_retry: недостижимая ветка")


def parse_retry_after(value: str | None) -> float | None:
    """Заголовок Retry-After в секундах. Формат с датой игнорируем: провайдеры
    моделей присылают секунды, а разбор HTTP-даты ради полноты не окупается."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
