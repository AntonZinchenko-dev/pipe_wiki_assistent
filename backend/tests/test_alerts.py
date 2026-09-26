"""Алерты: то, из-за чего их обычно и отключают.

Алерт легко написать и трудно сделать полезным. Ломается он четырьмя
способами, и все четыре кончаются одинаково — человек перестаёт их читать:

1. срабатывает на выборке из двух запросов и кричит после каждого
   перезапуска;
2. повторяет одно и то же сообщение на каждый запрос, забивая журнал;
3. не сообщает, что всё кончилось, и дежурный сидит над закончившейся
   аварией;
4. считает отказом нормальную работу — например, честный ответ «в
   документации этого нет».

Плюс пятое, самое обидное: порог, при котором алерт горит всегда.
"""

from __future__ import annotations

from app.alerts import AlertMonitor, Level
from app.config import Settings
from app.trace import Trace


class Clock:
    """Управляемые часы.

    Всё поведение монитора зависит от времени: скользящее окно, суточный
    бюджет, возврат в норму по мере старения ошибок. Проверять это
    ожиданием пятнадцати минут невозможно, поэтому время двигаем руками —
    и заодно проверяем, что окно вообще очищается.
    """

    def __init__(self) -> None:
        self.at = 1_790_000_000.0

    def __call__(self) -> float:
        return self.at

    def forward(self, minutes: float) -> None:
        self.at += minutes * 60


def monitor(**overrides) -> AlertMonitor:
    base = {"alert_min_samples": 5, "alert_window_minutes": 15}
    base.update(overrides)
    return AlertMonitor(Settings(**base))


def monitor_with_clock(**overrides) -> tuple[AlertMonitor, Clock]:
    base = {"alert_min_samples": 5, "alert_window_minutes": 15}
    base.update(overrides)
    clock = Clock()
    return AlertMonitor(Settings(**base), now=clock), clock


def trace(*, status="answered", total_ms=1000.0, cost=0.0, failed_citations=0) -> Trace:
    item = Trace(trace_id="x", question="вопрос")
    item.status = status
    item.total_ms = total_ms
    item.cost_rub = cost
    item.citations_failed = failed_citations
    return item


def reading(alerts: AlertMonitor, metric: str):
    return next(item for item in alerts.readings() if item.metric == metric)


# ------------------------------------------------------- пороги и источник


def test_thresholds_match_the_regulation() -> None:
    """Пороги взяты из РЛ-9.2, а не сочинены.

    Дежурный работает по регламенту. Система, оповещающая по другим числам,
    ссорит его с регламентом — и спор о том, авария это или нет, начинается
    в момент аварии.
    """
    settings = Settings()
    assert settings.alert_error_rate_warn == 0.01
    assert settings.alert_error_rate_crit == 0.05
    assert settings.alert_budget_warn == 0.80
    assert settings.alert_budget_crit == 1.20


def test_latency_thresholds_deliberately_differ_and_say_so() -> None:
    """У задержки свои значения, и расхождение названо вслух.

    В регламенте «1,5 с внимание, 4 с авария» написано про API данных. Для
    генерации это означало бы вечно горящий алерт, а вечно горящий алерт
    отключают вместе со всеми остальными.
    """
    settings = Settings()
    assert settings.alert_latency_warn_s > 4.0
    assert "API данных" in reading(monitor(), "latency_p95").source


# ---------------------------------------------------------------- выборка


def test_no_alarm_on_a_tiny_sample() -> None:
    """Одна ошибка из двух запросов — это 50 %, и это ничего не значит.

    Без нижней границы выборки алерт срабатывает на каждом перезапуске: с
    утра пришёл один человек, у него не сошлось, и система объявляет аварию.
    """
    alerts = monitor(alert_min_samples=5)
    alerts.observe(trace(status="internal_error"))
    alerts.observe(trace())

    item = reading(alerts, "error_rate")
    assert item.level is Level.OK
    assert item.value is None
    # И честно сказано, почему молчим, а не «всё хорошо».
    assert "нужно 5" in item.detail


# --------------------------------------------------------------- переходы


def test_the_alert_fires_once_not_on_every_request() -> None:
    """Иначе журнал забивается одинаковыми строками и его перестают читать.

    Это и есть настоящая причина пропущенных аварий: не отсутствие алерта,
    а привычка его пролистывать.
    """
    alerts = monitor(alert_min_samples=5)
    for _ in range(10):
        alerts.observe(trace(status="provider_error"))

    events = [event for event in alerts.report()["events"] if event["metric"] == "error_rate"]
    assert len(events) == 1
    assert events[0]["level"] == "crit"


def test_recovery_is_an_event_too() -> None:
    """Без этого по журналу не понять, кончилось ли.

    Дежурный, увидевший утром аварию без отбоя, идёт разбирать проблему,
    которой уже нет.

    Гасится алерт СТАРЕНИЕМ ошибок, а не тем, что их «перевесили»
    успешными запросами. Это разные механизмы: первый работает и на
    системе, куда за ночь пришло два человека, второй — только на
    нагруженной. Заодно проверяем, что окно вообще очищается: окно,
    которое не очищается, — это алерт, который не гаснет никогда.
    """
    alerts, clock = monitor_with_clock(alert_min_samples=5, alert_window_minutes=15)
    for _ in range(10):
        alerts.observe(trace(status="provider_error"))

    events = [event for event in alerts.report()["events"] if event["metric"] == "error_rate"]
    assert events[-1]["level"] == "crit"

    # Прошло двадцать минут, ошибки вышли из окна, пришло пять нормальных.
    clock.forward(20)
    for _ in range(5):
        alerts.observe(trace())

    events = [event for event in alerts.report()["events"] if event["metric"] == "error_rate"]
    assert events[-1]["level"] == "ok"
    assert "вернулось в норму" in events[-1]["text"]


def test_the_window_actually_forgets() -> None:
    """Старые запросы обязаны выпадать из расчёта.

    Иначе метрика — это не «что происходит сейчас», а «что происходило с
    момента запуска», и после одного плохого часа она врёт до конца недели.
    """
    alerts, clock = monitor_with_clock(alert_min_samples=1, alert_window_minutes=15)
    for _ in range(5):
        alerts.observe(trace(status="provider_error"))
    assert reading(alerts, "error_rate").value == 1.0

    clock.forward(16)
    assert reading(alerts, "error_rate").value is None


# ------------------------------------------------------- что считать сбоем


def test_a_documented_absence_of_an_answer_is_not_a_failure() -> None:
    """«В документации этого нет» — правильная работа, а не поломка.

    Смешать их значило бы поднимать аварию каждый раз, когда кто-то
    спросил про то, чего в вики нет. А таких вопросов у нас в эталонном
    наборе тринадцать штук специально.
    """
    alerts = monitor(alert_min_samples=5)
    for _ in range(20):
        alerts.observe(trace(status="not_found"))

    assert reading(alerts, "error_rate").level is Level.OK
    assert reading(alerts, "error_rate").value == 0.0


# ------------------------------------------------------------- процентиль


def test_the_tail_is_visible_because_it_is_a_percentile() -> None:
    """Среднее спрятало бы один тридцатисекундный ответ за девятью быстрыми.

    Регламент требует 95-й процентиль ровно по этой причине: жалуется тот,
    кто попал в хвост, а не тот, кто попал в среднее.
    """
    alerts = monitor(alert_min_samples=5, alert_latency_warn_s=8, alert_latency_crit_s=20)
    for _ in range(19):
        alerts.observe(trace(total_ms=900))
    alerts.observe(trace(total_ms=40_000))

    item = reading(alerts, "latency_p95")
    # Среднее здесь было бы около 2,9 с — то есть «всё хорошо».
    assert item.value is not None and item.value >= 20
    assert item.level is Level.CRITICAL


# ----------------------------------------------------------------- бюджет


def test_an_unset_budget_is_not_zero_percent_spent() -> None:
    """Ноль процентов от незаданного бюджета читался бы как «всё хорошо».

    А означает он «не с чем сравнить». Разница та же, что между «потратили
    ноль» и «не посчитали» — на ней мы уже обжигались со стоимостью
    облачных ответов.
    """
    alerts = monitor(daily_cost_budget_rub=0.0)
    alerts.observe(trace(cost=12.5))

    item = reading(alerts, "daily_budget")
    assert item.value is None
    assert item.level is Level.OK
    assert "не задан" in item.detail
    # Но сам расход показан: молчать про потраченные деньги нельзя.
    assert "12.50" in item.detail


def test_spending_over_budget_raises_the_regulation_level() -> None:
    alerts = monitor(daily_cost_budget_rub=100.0)
    for _ in range(9):
        alerts.observe(trace(cost=10.0))
    assert reading(alerts, "daily_budget").level is Level.WARN

    for _ in range(4):
        alerts.observe(trace(cost=10.0))
    assert reading(alerts, "daily_budget").level is Level.CRITICAL


# --------------------------------------------------- достоверность ответов


def test_unverified_citations_are_watched_separately() -> None:
    """Недостоверность — инцидент тяжелее недоступности, так говорит регламент.

    Упавший сервис виден сразу; неверное число уходит в отчёт заказчику и
    обнаруживается через недели. Поэтому у доли неподтверждённых ссылок
    свой порог, а не общий с отказами.
    """
    alerts = monitor(alert_min_samples=5, alert_unverified_warn=0.2, alert_unverified_crit=0.4)
    for _ in range(10):
        alerts.observe(trace(failed_citations=1))

    item = reading(alerts, "unverified_citations")
    assert item.level is Level.CRITICAL
    assert item.value == 1.0


def test_the_report_carries_thresholds_next_to_values() -> None:
    """Цифра без порога не значит ничего.

    «0.07» требует лезть в регламент. «0.07 при пороге внимания 0.01»
    отвечает на вопрос сразу — а отвечать на него приходится ночью.
    """
    alerts = monitor(alert_min_samples=1)
    alerts.observe(trace())

    for item in alerts.report()["metrics"]:
        assert "warn_at" in item and "crit_at" in item
        assert item["source"], item["metric"]
