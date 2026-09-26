"""Калибровка абсолютного порога отсечения.

Порог 0.45 в конфигурации был взят на глаз — это единственное необоснованное
число во всей системе, и именно такие числа потом живут годами. Здесь оно
заменяется измеренным.

Что измеряем. Порог решает одну задачу: отличить «в корпусе есть ответ» от
«в корпусе ответа нет». Значит у него есть ровно две ошибки, и они тянут в
разные стороны:

- порог слишком высокий -> отвечаемые вопросы получают отказ (потеря пользы);
- порог слишком низкий -> неотвечаемые получают контекст из мусора, и модель
  на нём сочиняет (потеря доверия).

Поэтому калибровка — это не поиск «лучшего числа», а выбор точки на кривой
компромисса, и выбирать её надо осознанно. Для справочной системы по
регламентам цена выдумки выше цены отказа, поэтому по умолчанию мы берём
точку, где отказ по неотвечаемым близок к максимуму, а потери по отвечаемым
ещё приемлемы.

Важно: калибровать порог можно только по КОСИНУСУ. У RRF нет физического
смысла, его значение зависит только от позиций в списках и всегда «нормальное».
"""

from __future__ import annotations

from dataclasses import dataclass

from .runner import RowResult


@dataclass(slots=True)
class ThresholdPoint:
    floor: float
    answerable_passed: float     # доля отвечаемых, прошедших порог (хотим высоко)
    unanswerable_rejected: float  # доля неотвечаемых, отсечённых (хотим высоко)
    balanced: float               # среднее двух — одна цифра для сортировки
    answerable_lost: int          # сколько отвечаемых вопросов потеряли
    unanswerable_leaked: int      # сколько неотвечаемых просочилось

    def as_dict(self) -> dict:
        return {
            "floor": round(self.floor, 3),
            "отвечаемые прошли": round(self.answerable_passed, 3),
            "неотвечаемые отсечены": round(self.unanswerable_rejected, 3),
            "баланс": round(self.balanced, 3),
            "потеряно отвечаемых": self.answerable_lost,
            "просочилось неотвечаемых": self.unanswerable_leaked,
        }


def sweep(rows: list[RowResult], *, start: float = 0.20, stop: float = 0.75, step: float = 0.01) -> list[ThresholdPoint]:
    """Развёртка порога по сохранённому поисковому прогону.

    Ключевая экономия: прогон делается ОДИН раз, а пороги перебираются по
    сохранённым значениям косинуса. Гонять поиск заново на каждый порог не
    нужно — сам поиск от порога не зависит, порог только отсекает результат.
    """
    answerable = [row for row in rows if row.answerable]
    unanswerable = [row for row in rows if not row.answerable]
    points: list[ThresholdPoint] = []

    value = start
    while value <= stop + 1e-9:
        passed = [row for row in answerable if row.retrieval["best_cosine"] >= value]
        leaked = [row for row in unanswerable if row.retrieval["best_cosine"] >= value]

        answerable_passed = len(passed) / len(answerable) if answerable else 0.0
        unanswerable_rejected = (
            1 - len(leaked) / len(unanswerable) if unanswerable else 0.0
        )
        points.append(
            ThresholdPoint(
                floor=value,
                answerable_passed=answerable_passed,
                unanswerable_rejected=unanswerable_rejected,
                balanced=(answerable_passed + unanswerable_rejected) / 2,
                answerable_lost=len(answerable) - len(passed),
                unanswerable_leaked=len(leaked),
            )
        )
        value = round(value + step, 6)

    return points


def recommend(points: list[ThresholdPoint], *, min_answerable: float = 0.95) -> ThresholdPoint:
    """Рекомендация: максимум отсечения мусора при заданной сохранности пользы.

    `min_answerable` — это продуктовое решение, а не математическое. «Не
    потерять больше 5 % отвечаемых вопросов» — то, с чем должен согласиться
    владелец продукта; всё остальное считается автоматически.
    """
    eligible = [point for point in points if point.answerable_passed >= min_answerable]
    if not eligible:
        # Требование недостижимо ни при каком пороге: честно отдаём лучший по
        # балансу и оставляем решение человеку.
        return max(points, key=lambda point: point.balanced)
    return max(eligible, key=lambda point: (point.unanswerable_rejected, point.floor))


def render(points: list[ThresholdPoint], *, recommended: ThresholdPoint, width: int = 42) -> str:
    """Текстовая кривая компромисса.

    График здесь полезнее таблицы: видно не только выбранную точку, но и
    насколько она устойчива — то есть далеко ли до обрыва, где начинают
    теряться отвечаемые вопросы.
    """
    lines = [
        f"{'порог':>6}  {'отвечаемые':>10}  {'отказы':>7}  кривая",
    ]
    for point in points:
        if round(point.floor * 100) % 5:  # печатаем каждые 0.05, чтобы читалось
            continue
        answer_bar = "#" * round(point.answerable_passed * width)
        reject_bar = "·" * round(point.unanswerable_rejected * width)
        mark = " <- рекомендация" if abs(point.floor - recommended.floor) < 1e-9 else ""
        lines.append(
            f"{point.floor:>6.2f}  {point.answerable_passed:>10.2f}  "
            f"{point.unanswerable_rejected:>7.2f}  {answer_bar}|{reject_bar}{mark}"
        )
    lines.append("")
    lines.append("# — доля отвечаемых вопросов, прошедших порог (польза)")
    lines.append("· — доля неотвечаемых вопросов, отсечённых (защита от выдумок)")
    return "\n".join(lines)
