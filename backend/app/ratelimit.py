"""Ограничение частоты на нашей стороне — по запросам И по токенам.

Лимит только по числу запросов не защищает от одного тяжёлого: запрос с
огромным контекстом стоит провайдеру как двадцать обычных. Поэтому окно
считает обе величины (раздел 2).

Состояние в памяти процесса: для одного инстанса достаточно, при нескольких
переедет в Redis без изменения интерфейса.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Window:
    events: deque[tuple[float, int]] = field(default_factory=deque)

    def prune(self, now: float, span_s: float) -> None:
        while self.events and now - self.events[0][0] > span_s:
            self.events.popleft()

    def totals(self) -> tuple[int, int]:
        return len(self.events), sum(tokens for _, tokens in self.events)


@dataclass(slots=True)
class Decision:
    allowed: bool
    retry_after_s: int = 0
    reason: str = ""


class RateLimiter:
    def __init__(self, *, rpm: int, tpm: int, span_s: float = 60.0) -> None:
        self._rpm = rpm
        self._tpm = tpm
        self._span = span_s
        self._windows: dict[str, Window] = {}

    def check(self, key: str, *, estimated_tokens: int) -> Decision:
        now = time.monotonic()
        window = self._windows.setdefault(key, Window())
        window.prune(now, self._span)
        requests, tokens = window.totals()

        def wait_hint() -> int:
            """Через сколько освободится место. При пустом окне — целое окно.

            Раньше здесь безусловно читался `window.events[0]`, и на ПУСТОМ
            окне это был IndexError: такое возможно, когда оценка одного
            запроса сама больше лимита токенов. Отказ по лимиту превращался во
            внутреннюю ошибку 500 — то есть защита от перегрузки становилась
            источником ошибки.
            """
            if not window.events:
                return int(self._span)
            return max(1, int(self._span - (now - window.events[0][0])))

        if requests >= self._rpm:
            # Сообщение об отказе конкретное: «через 40 секунд», а не «слишком
            # много запросов» — человек должен знать, ждать ему секунду или час
            # (раздел 2).
            return Decision(False, wait_hint(), "лимит запросов")

        if tokens + estimated_tokens > self._tpm:
            if not window.events:
                # Окно пусто, а запрос всё равно не проходит: он один тяжелее
                # всего лимита. Ждать бессмысленно — сколько бы ни ждали,
                # ничего не изменится. Честнее сказать это прямо.
                return Decision(
                    False, int(self._span),
                    "запрос тяжелее всего лимита токенов — ожидание не поможет",
                )
            return Decision(False, wait_hint(), "лимит токенов")

        window.events.append((now, estimated_tokens))
        return Decision(True)


# Заголовки, которым МОЖНО верить, только если мы точно за своим прокси.
# Список закрытый и короткий: каждый заголовок здесь — это доверие к тому, что
# его выставил наш собственный балансировщик, а не пользователь.
FORWARDED_HEADERS = ("x-forwarded-for", "x-real-ip")


def client_key(headers, peer: str | None, *, trust_proxy: bool) -> str:
    """Ключ, по которому считается лимит одного клиента.

    Дефект, который это лечит, тихий и полностью меняет смысл лимитера.

    За обратным прокси (nginx, ingress, Cloudflare — то есть в любом
    развёртывании, кроме локального) адрес соединения — это адрес ПРОКСИ, один
    и тот же для всех. Лимитер в такой схеме считает не «тридцать запросов в
    минуту на пользователя», а «тридцать на всех вместе»: один активный
    человек исчерпывает квоту всей компании, и остальные видят 429 на пустой
    системе. Причём в логах всё выглядит нормально — лимитер исправно
    отработал.

    Обратная ошибка не менее опасна и потому вынесена в отдельный флаг. Если
    доверять X-Forwarded-For БЕЗ прокси впереди, то заголовок ставит сам
    клиент, и обойти лимит можно, меняя в нём одну цифру. Лимитер тогда не
    ослаблен, а выключен — при полной видимости работы.

    Поэтому доверие включается настройкой явно и по умолчанию выключено:
    безопасное значение — то, которое ничего не ломает в незнакомой схеме.

    Аналогия: охранник на входе записывает номер машины. Если все приезжают на
    одном служебном автобусе, в журнале будет один номер на сотню человек —
    и лимит «один вход на номер» запретит вход всем, кроме первого. А если
    верить тому номеру, который пассажир назвал сам, журнал перестанет
    что-либо значить вовсе.

    Берётся ПЕРВЫЙ адрес из списка: прокси дописывают свои адреса справа,
    поэтому левый — исходный клиент.
    """
    if trust_proxy:
        for header in FORWARDED_HEADERS:
            value = headers.get(header)
            if value:
                first = value.split(",")[0].strip()
                if first:
                    return first
    return peer or "unknown"
