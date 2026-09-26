"""Диагностика разделимости: можно ли вообще отличить «ответа нет» по числам.

Этот файл появился из результата первого настоящего прогона. Картина была
такая: поиск находит нужный документ в 100 % отвечаемых вопросов, а из десяти
вопросов, ответа на которые в корпусе нет, порог 0.45 не отсекает ни одного.
То есть на «сколько стоит подписка» система находит что-то достаточно похожее
и идёт генерировать ответ.

Чек-лист требует проверить модель эмбеддингов на своём языке: взять верные
пары и правдоподобные, но неверные, и посмотреть, не налезают ли
распределения. Здесь это делается на живых данных прогона, а не на
искусственных парах — и налезают они сильно: у bge-m3 нет «нулевой» точки, он
выдаёт высокие косинусы и нерелевантному тексту.

Отсюда вопрос, на который отвечает этот модуль: существует ли вообще числовой
признак, разделяющий отвечаемые и неотвечаемые вопросы лучше, чем абсолютный
косинус. Проверяются два кандидата:

- `best_cosine` — как сейчас;
- `margin` — насколько первый результат выделяется на фоне следующих.

Если разделимость плохая у обоих, вывод честный и важный: порог — слабый
рубеж, и решение «есть ли ответ» должна принимать модель, видя контекст. Порог
тогда остаётся дешёвым предфильтром явного мусора, а не главной защитой.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(slots=True)
class Distribution:
    name: str
    values: list[float]

    @property
    def summary(self) -> dict:
        if not self.values:
            return {}
        ordered = sorted(self.values)
        return {
            "n": len(ordered),
            "min": round(ordered[0], 3),
            "p25": round(ordered[len(ordered) // 4], 3),
            "median": round(statistics.median(ordered), 3),
            "p75": round(ordered[3 * len(ordered) // 4], 3),
            "max": round(ordered[-1], 3),
        }


@dataclass(slots=True)
class Separability:
    feature: str
    best_threshold: float
    answerable_kept: float
    unanswerable_rejected: float
    balanced: float
    overlap: float  # доля неотвечаемых, попавших в диапазон отвечаемых

    def as_dict(self) -> dict:
        return {
            "признак": self.feature,
            "лучший порог": round(self.best_threshold, 3),
            "сохранено пользы": round(self.answerable_kept, 3),
            "отсечено мусора": round(self.unanswerable_rejected, 3),
            "баланс": round(self.balanced, 3),
            "перекрытие": round(self.overlap, 3),
        }


def separability(answerable: list[float], unanswerable: list[float], *, feature: str) -> Separability:
    """Лучшее, чего можно добиться этим признаком в принципе.

    Перебираем все пороги, которые вообще имеет смысл пробовать (сами
    наблюдённые значения), и берём точку максимального баланса. Это верхняя
    граница возможностей признака: если даже она низкая, признак не годится, и
    крутить порог бессмысленно.
    """
    if not answerable or not unanswerable:
        return Separability(feature, 0.0, 0.0, 0.0, 0.0, 1.0)

    candidates = sorted(set(answerable) | set(unanswerable))
    best = Separability(feature, 0.0, 0.0, 0.0, -1.0, 1.0)

    for threshold in candidates:
        kept = sum(1 for value in answerable if value >= threshold) / len(answerable)
        rejected = sum(1 for value in unanswerable if value < threshold) / len(unanswerable)
        balanced = (kept + rejected) / 2
        if balanced > best.balanced:
            best = Separability(feature, threshold, kept, rejected, balanced, 0.0)

    low, high = min(answerable), max(answerable)
    best.overlap = sum(1 for value in unanswerable if low <= value <= high) / len(unanswerable)
    return best


def analyze(rows: list[dict]) -> str:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]

    lines: list[str] = []
    lines.append(
        f"отвечаемых вопросов {len(answerable)}, неотвечаемых {len(unanswerable)}"
    )
    lines.append("")

    features = {
        "best_cosine": "лучший косинус",
        "margin": "разрыв между первым и следующими",
    }

    results: list[Separability] = []
    for key, title in features.items():
        yes = [float(row["retrieval"].get(key, 0.0)) for row in answerable]
        no = [float(row["retrieval"].get(key, 0.0)) for row in unanswerable]
        if not any(yes) and not any(no):
            continue

        lines.append(f"{title} ({key})")
        lines.append(f"  есть ответ:  {Distribution(key, yes).summary}")
        lines.append(f"  нет ответа:  {Distribution(key, no).summary}")
        result = separability(yes, no, feature=key)
        results.append(result)
        lines.append(f"  разделимость: {result.as_dict()}")
        lines.append("")

    if results:
        winner = max(results, key=lambda item: item.balanced)
        lines.append(f"лучший признак: {winner.feature} (баланс {winner.balanced:.3f})")
        if winner.balanced < 0.85:
            lines.append(
                "  Это НЕ достаточно для самостоятельного рубежа. Вывод по чек-листу\n"
                "  прямой: распределения налезают, и промпт этого не исправит. Порог\n"
                "  остаётся дешёвым предфильтром явного мусора, а решение «есть ли\n"
                "  ответ» принимает модель, видя контекст, — и её долю правильных\n"
                "  отказов надо мерить прогоном answer, а не поиском."
            )
        else:
            lines.append(
                "  Признак пригоден как рубеж: можно ставить порог по нему и мерить\n"
                "  результат отдельным прогоном."
            )
        lines.append("")

    ranks = [row["retrieval"].get("chunk_rank", 0) for row in answerable]
    found = [rank for rank in ranks if rank]
    if found:
        first = sum(1 for rank in found if rank == 1) / len(answerable)
        top3 = sum(1 for rank in found if rank <= 3) / len(answerable)
        lines.append("позиция нужного ЧАНКА в выдаче (не насыщается, в отличие от recall)")
        lines.append(f"  первым:        {first:.3f}")
        lines.append(f"  в первых трёх: {top3:.3f}")
        lines.append(f"  не найден:     {(len(answerable) - len(found)) / len(answerable):.3f}")
        lines.append(
            f"  медианная позиция: {statistics.median(found):.1f}, худшая: {max(found)}"
        )
        lines.append("")
        worst = sorted(
            (row for row in answerable if row["retrieval"].get("chunk_rank", 0) > 3),
            key=lambda row: -row["retrieval"]["chunk_rank"],
        )
        if worst:
            lines.append(f"нужный фрагмент стоял ниже третьего места ({len(worst)}):")
            for row in worst[:10]:
                lines.append(
                    f"  {row['question_id']}  позиция {row['retrieval']['chunk_rank']:>2}  "
                    f"{row['question'][:52]}"
                )
            lines.append(
                "  Это цель для реранкера: состав выдачи менять не нужно, нужен порядок."
            )

    return "\n".join(lines)
