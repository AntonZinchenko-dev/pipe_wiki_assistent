"""Алерты: превышение порогов замечает система, а не дежурный.

Лимиты и таймауты в проекте были и раньше. Не было главного — чтобы кто-то
заметил, что их пробили. Лимит без наблюдения защищает пользователя от
перегрузки, но не говорит вам, что перегрузка случилась; вы узнаёте об этом
из жалобы.

ПОРОГИ ВЗЯТЫ ИЗ РЕГЛАМЕНТА, А НЕ ПРИДУМАНЫ

В вики есть таблица порогов оповещения (РЛ-9.2, «Пороги оповещения»), и два
из них применимы к нам напрямую: доля ошибок и расход от дневного бюджета.
Брать их оттуда, а не сочинять свои, — не формальность: дежурный работает по
регламенту, и система, оповещающая по другим числам, ссорит его с
регламентом.

ДВА ПОРОГА ИЗ РЕГЛАМЕНТА МЫ СОЗНАТЕЛЬНО НЕ ПРИМЕНЯЕМ КАК ЕСТЬ

Задержка «1,5 с внимание, 4 с авария» в регламенте написана про API данных.
Для генерации текста это бессмысленные числа: нормальный ответ модели идёт
дольше четырёх секунд, и такой порог кричал бы постоянно. Алерт, который
срабатывает всегда, отключают через неделю — и вместе с ним отключают все
остальные. Поэтому у задержки генерации свои значения, и они в настройках, а
не в коде.

ЧЕТЫРЕ ПРАВИЛА, БЕЗ КОТОРЫХ АЛЕРТЫ ВРЕДНЫ

1. ПРОЦЕНТИЛЬ, А НЕ СРЕДНЕЕ. Среднее прячет хвост: девять быстрых ответов и
   один тридцатисекундный дают приличное среднее и одного злого человека.
   Регламент тоже требует 95-й процентиль.
2. МИНИМУМ НАБЛЮДЕНИЙ. Одна ошибка из двух запросов — это 50 %, и это ничего
   не значит. Без нижней границы выборки алерт срабатывает на каждом
   перезапуске.
3. СРАБАТЫВАНИЕ НА ПЕРЕХОДЕ, А НЕ НА КАЖДОМ ЗАПРОСЕ. Иначе журнал забивается
   одинаковыми строками, и их перестают читать — это и есть настоящая
   причина пропущенных аварий.
4. ВОЗВРАТ В НОРМУ ТОЖЕ СОБЫТИЕ. Без него по журналу невозможно понять,
   кончилось ли, и дежурный сидит до утра над закончившейся проблемой.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from .config import Settings
from .trace import Trace

log = logging.getLogger("pipewiki.alerts")

# Статусы, которые считаем отказом системы.
#
# Отказ модели найти ответ (`not_found`) сюда НЕ входит, и это важно: «в
# документации нет ответа» — правильная работа системы, а не её поломка.
# Смешать их значило бы поднимать аварию каждый раз, когда кто-то спросил
# про то, чего в вики нет.
FAILED_STATUSES = frozenset({
    "internal_error", "provider_error", "provider_unavailable",
    "embed_unavailable", "index_mismatch",
})


class Level(str, Enum):
    OK = "ok"
    WARN = "warn"       # «порог внимания» по регламенту
    CRITICAL = "crit"   # «порог аварии» по регламенту


@dataclass(slots=True)
class Reading:
    """Показание одной метрики: значение, уровень и пороги рядом.

    Пороги отдаются вместе со значением сознательно. Цифра «0.07» сама по
    себе не говорит ничего; «0.07 при пороге внимания 0.01» говорит всё, и
    дежурному не надо искать регламент.
    """

    metric: str
    title: str
    value: float | None
    warn_at: float
    crit_at: float
    level: Level
    detail: str = ""
    source: str = ""


@dataclass(slots=True)
class AlertEvent:
    at: float
    metric: str
    level: Level
    previous: Level
    value: float | None
    text: str


@dataclass(slots=True)
class _Sample:
    at: float
    failed: bool
    total_ms: float
    cost_rub: float
    citations_failed: int
    answered: bool


class AlertMonitor:
    """Считает метрики по последним запросам и поднимает алерты на переходах."""

    def __init__(self, settings: Settings, now=time.time) -> None:
        # Часы передаются снаружи, и это не украшение теста.
        #
        # Всё поведение здесь зависит от времени: скользящее окно, суточный
        # бюджет, возврат в норму по мере старения ошибок. Компонент, у
        # которого время нельзя подвинуть, — это компонент, чья временная
        # логика не проверена ничем, кроме ожидания пятнадцати минут. Именно
        # там и живут ошибки: окно, которое не очищается, и алерт, который
        # никогда не гаснет.
        self._now = now
        self._s = settings
        self._samples: deque[_Sample] = deque(maxlen=2000)
        self._levels: dict[str, Level] = {}
        self._events: deque[AlertEvent] = deque(maxlen=100)
        # Расход считается за сутки, а не за окно наблюдения: бюджет в
        # регламенте дневной. Ключ — календарная дата, чтобы счётчик сам
        # обнулялся в полночь и не тянул вчерашний перерасход в сегодня.
        self._spent: dict[str, float] = {}

    # ------------------------------------------------------------ наблюдение

    def observe(self, trace: Trace) -> None:
        """Учесть завершившийся запрос. Вызывается на каждом ответе."""
        now = self._now()
        self._samples.append(_Sample(
            at=now,
            failed=trace.status in FAILED_STATUSES or bool(trace.error),
            total_ms=trace.total_ms,
            cost_rub=trace.cost_rub,
            citations_failed=trace.citations_failed,
            answered=trace.status == "answered",
        ))
        if trace.cost_rub:
            day = time.strftime("%Y-%m-%d", time.localtime(now))
            self._spent[day] = self._spent.get(day, 0.0) + trace.cost_rub

        for reading in self.readings():
            self._transition(reading)

    def _transition(self, reading: Reading) -> None:
        """Событие возникает на СМЕНЕ уровня, а не на каждом запросе."""
        previous = self._levels.get(reading.metric, Level.OK)
        if reading.level is previous:
            return
        self._levels[reading.metric] = reading.level

        if reading.level is Level.OK:
            text = f"вернулось в норму: {reading.title} — {reading.detail}"
            log.info("АЛЕРТ СНЯТ %s: %s", reading.metric, text)
        else:
            marker = "ВНИМАНИЕ" if reading.level is Level.WARN else "АВАРИЯ"
            text = f"{marker}: {reading.title} — {reading.detail}"
            # Уровень записи разный: авария обязана быть видна в журнале
            # отдельно от предупреждений, иначе фильтровать нечем.
            (log.error if reading.level is Level.CRITICAL else log.warning)(
                "АЛЕРТ %s: %s", reading.metric, text
            )

        self._events.append(AlertEvent(
            at=self._now(), metric=reading.metric, level=reading.level,
            previous=previous, value=reading.value, text=text,
        ))

    # -------------------------------------------------------------- метрики

    def _window(self) -> list[_Sample]:
        edge = self._now() - self._s.alert_window_minutes * 60
        return [sample for sample in self._samples if sample.at >= edge]

    def readings(self) -> list[Reading]:
        window = self._window()
        return [
            self._error_rate(window),
            self._latency(window),
            self._citations(window),
            self._budget(),
        ]

    def _level(self, value: float | None, warn: float, crit: float) -> Level:
        if value is None:
            return Level.OK
        if crit and value >= crit:
            return Level.CRITICAL
        if warn and value >= warn:
            return Level.WARN
        return Level.OK

    def _enough(self, window: list[_Sample]) -> bool:
        return len(window) >= self._s.alert_min_samples

    def _error_rate(self, window: list[_Sample]) -> Reading:
        warn, crit = self._s.alert_error_rate_warn, self._s.alert_error_rate_crit
        if not self._enough(window):
            return Reading(
                "error_rate", "доля отказов системы", None, warn, crit, Level.OK,
                detail=f"наблюдений {len(window)}, нужно {self._s.alert_min_samples}",
                source="РЛ-9.2, доля ошибок 5xx",
            )
        value = sum(1 for sample in window if sample.failed) / len(window)
        return Reading(
            "error_rate", "доля отказов системы", round(value, 4), warn, crit,
            self._level(value, warn, crit),
            detail=f"{value:.1%} за {self._s.alert_window_minutes} мин "
                   f"(порог внимания {warn:.0%}, аварии {crit:.0%})",
            source="РЛ-9.2, доля ошибок 5xx",
        )

    def _latency(self, window: list[_Sample]) -> Reading:
        warn, crit = self._s.alert_latency_warn_s, self._s.alert_latency_crit_s
        if not self._enough(window):
            return Reading(
                "latency_p95", "задержка ответа, 95-й процентиль", None, warn, crit,
                Level.OK,
                detail=f"наблюдений {len(window)}, нужно {self._s.alert_min_samples}",
                source="РЛ-9.2, но свои значения: там порог про API данных",
            )
        value = _percentile([sample.total_ms / 1000 for sample in window], 0.95)
        return Reading(
            "latency_p95", "задержка ответа, 95-й процентиль", round(value, 2),
            warn, crit, self._level(value, warn, crit),
            detail=f"{value:.1f} с (порог внимания {warn:.0f} с, аварии {crit:.0f} с)",
            source="РЛ-9.2, но свои значения: там порог про API данных",
        )

    def _citations(self, window: list[_Sample]) -> Reading:
        """Доля ответов, в которых не подтвердилась ни одна ссылка.

        Метрика НАША, не из регламента, и она про недостоверность, а не про
        недоступность. Регламент прямо говорит, что недостоверные данные —
        инцидент более тяжёлый, чем недоступность: упавший сервис виден
        сразу, а неверное число уходит в отчёт заказчику и обнаруживается
        через недели. Рост этой доли — первый признак, что поиск или промпт
        поехали, и ловится он раньше жалоб.
        """
        warn, crit = self._s.alert_unverified_warn, self._s.alert_unverified_crit
        answered = [sample for sample in window if sample.answered]
        if len(answered) < self._s.alert_min_samples:
            return Reading(
                "unverified_citations", "ответы с неподтверждёнными ссылками",
                None, warn, crit, Level.OK,
                detail=f"ответов {len(answered)}, нужно {self._s.alert_min_samples}",
                source="наша метрика достоверности",
            )
        value = sum(1 for sample in answered if sample.citations_failed) / len(answered)
        return Reading(
            "unverified_citations", "ответы с неподтверждёнными ссылками",
            round(value, 4), warn, crit, self._level(value, warn, crit),
            detail=f"{value:.1%} ответов (порог внимания {warn:.0%}, аварии {crit:.0%})",
            source="наша метрика достоверности",
        )

    def _budget(self) -> Reading:
        warn, crit = self._s.alert_budget_warn, self._s.alert_budget_crit
        budget = self._s.daily_cost_budget_rub
        spent = self._spent.get(time.strftime("%Y-%m-%d", time.localtime(self._now())), 0.0)
        if budget <= 0:
            # Бюджет не задан — считать долю не от чего. Показываем расход
            # и честно говорим, что порога нет: «0 %» здесь означало бы
            # «всё хорошо», а на самом деле означает «не с чем сравнить».
            return Reading(
                "daily_budget", "расход за сутки", None, warn, crit, Level.OK,
                detail=f"потрачено {spent:.2f} ₽, дневной бюджет не задан "
                       f"(PW_DAILY_COST_BUDGET_RUB)",
                source="РЛ-9.2, расход от дневного бюджета",
            )
        value = spent / budget
        return Reading(
            "daily_budget", "расход за сутки", round(value, 4), warn, crit,
            self._level(value, warn, crit),
            detail=f"{spent:.2f} ₽ из {budget:.2f} ₽ ({value:.0%}; "
                   f"порог внимания {warn:.0%}, аварии {crit:.0%})",
            source="РЛ-9.2, расход от дневного бюджета",
        )

    # --------------------------------------------------------------- отчёт

    def report(self) -> dict:
        readings = self.readings()
        worst = Level.OK
        for reading in readings:
            if reading.level is Level.CRITICAL:
                worst = Level.CRITICAL
            elif reading.level is Level.WARN and worst is Level.OK:
                worst = Level.WARN

        return {
            "level": worst.value,
            "window_minutes": self._s.alert_window_minutes,
            "samples": len(self._window()),
            "metrics": [
                {
                    "metric": reading.metric,
                    "title": reading.title,
                    "value": reading.value,
                    "level": reading.level.value,
                    "warn_at": reading.warn_at,
                    "crit_at": reading.crit_at,
                    "detail": reading.detail,
                    "source": reading.source,
                }
                for reading in readings
            ],
            # История переходов, а не текущих значений: по ней видно, когда
            # началось и когда кончилось. Текущее значение отвечает на вопрос
            # «что сейчас», история — на вопрос «что было ночью».
            "events": [
                {
                    "at": event.at,
                    "metric": event.metric,
                    "level": event.level.value,
                    "previous": event.previous.value,
                    "value": event.value,
                    "text": event.text,
                }
                for event in list(self._events)[-20:]
            ],
        }


def _percentile(values: list[float], share: float) -> float:
    """Процентиль по ближайшему рангу.

    Без интерполяции намеренно: на выборке в два десятка запросов
    интерполяция создаёт видимость точности, которой нет. Ближайший ранг
    всегда равен какому-то реально наблюдавшемуся значению — то есть числу,
    которое кто-то действительно прождал.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(share * len(ordered) + 0.5) - 1))
    return ordered[index]
