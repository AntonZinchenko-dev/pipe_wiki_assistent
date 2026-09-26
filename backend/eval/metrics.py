"""Метрики поиска и метрики ответа — раздельно.

Это не педантизм, а разные вопросы, и чинятся они по-разному:

- метрики ПОИСКА отвечают «нашли ли мы нужный кусок»;
- метрики ОТВЕТА отвечают «правильно ли модель им воспользовалась».

Если считать одну общую цифру, то на вопрос «почему стало хуже» ответа нет:
непонятно, поиск перестал находить или модель перестала пользоваться
найденным. Разрез по типам вопроса превращает одну бесполезную цифру в
диагноз ещё на шаг глубже: «просело на отрицаниях» или «просело на точных
кодах» — это уже готовая гипотеза.

Все метрики считаются на уровне ДОКУМЕНТОВ, а не чанков. Причина та же, что
и в разметке: номера чанков меняются при переиндексации, а `doc_id` — нет.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Доля нужных документов, попавших в первые k.

    Главная метрика поиска для нашей задачи: если нужного документа нет в
    выдаче, никакой промпт уже не поможет.
    """
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return len(relevant & set(top)) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Доля релевантных среди первых k.

    В нашей задаче она низкая по своей природе: релевантен обычно один
    документ из восьми в контексте. Смотреть на неё стоит не как на оценку
    «хорошо/плохо», а как на цену, которую мы платим токенами за recall.
    """
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    # Знаменатель — K, а не длина выдачи.
    #
    # При `min(k, len(top))` знаменателем всегда оказывалась сама длина
    # выдачи, то есть это была точность «по тому, что нашлось», а не
    # precision@k. В режиме ответа, где документов в контексте бывает два-три,
    # величина выходила завышенной в два-три раза относительно того же
    # показателя в поисковом прогоне — и два прогона нельзя было сравнивать,
    # хотя называется метрика одинаково.
    return round(len(relevant & set(top)) / k, 4)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Обратный ранг первого релевантного документа.

    Отвечает на вопрос «насколько высоко стоит правильный ответ». Для RAG это
    важнее precision: фрагмент с первой позиции модель использует охотнее, чем
    с восьмой, — середина контекста используется хуже краёв.
    """
    for position, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Нормированный DCG: единственная метрика здесь, чувствительная к ПОРЯДКУ.

    Именно поэтому она нужна, хотя выглядит избыточной рядом с recall.
    Реранкер не меняет СОСТАВ найденного, он меняет порядок — и recall его
    работу не заметит вообще. Без nDCG невозможно узнать, дал ли что-нибудь
    дорогой реранкер, который мы поставим на следующем шаге.
    """
    if not relevant:
        return 0.0
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved[:k]]
    dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def chunk_rank(chunk_texts: list[str], must_contain: list[str]) -> int:
    """Позиция первого чанка, реально содержащего ожидаемый текст. 0 — не нашёлся.

    Зачем нужна отдельная метрика на уровне чанков, когда есть recall по
    документам. Потому что на первом же настоящем прогоне recall по документам
    оказался ровно 1.000 по всем типам вопросов: документов всего шестнадцать,
    а в выдачу попадает четверть индекса, и «нужный документ где-то в топе»
    перестаёт быть достижением. Насыщенная метрика не измеряет ничего: она не
    покажет ни улучшения, ни ухудшения, пока всё не сломается целиком.

    Позиция нужного ЧАНКА не насыщается: она различает «фрагмент стоял
    первым» и «фрагмент был восьмым», а это именно та разница, которую делает
    реранкер и которую видит модель — середину контекста она использует хуже
    краёв.
    """
    if not must_contain:
        return 0
    needles = [" ".join(needle.split()).lower() for needle in must_contain]
    for position, text in enumerate(chunk_texts, start=1):
        haystack = " ".join(text.split()).lower()
        if all(needle in haystack for needle in needles):
            return position
    return 0


def answer_contains(answer: str, must_contain: list[str]) -> bool | None:
    """Самая дешёвая проверка правильности ОТВЕТА: есть ли в нём ключевой факт.

    Зачем она, если есть судья-модель. Судья стоит вызова, требует калибровки
    по каппе и сам врёт. А половина настоящих провалов выглядит грубо: модель
    написала «конкретный срок не указан» там, где во фрагменте стоит «не менее
    суток». Такое видно подстрокой, без вызова модели и без разметки.

    Эта проверка СТРОГАЯ В ОДНУ СТОРОНУ: `False` — сильный сигнал (факта в
    ответе нет), `True` — слабый (факт упомянут, но вывод мог быть кривым).
    Поэтому она не заменяет судью, а задаёт ему нижнюю границу: если
    `answer_contains` ниже `judge_ok`, судья добрый и его надо перепроверять.

    `None` значит «нечем проверять»: у вопроса нет обязательных подстрок или
    он неотвечаемый.

    Внутри одной подстроки допускается `|` — это «или»: ответ законно передаёт
    один и тот же факт разными словами, и требовать ровно нашей формулировки
    значит ловить ложные тревоги на правильных ответах. Список элементов —
    это «и»: каждый факт обязан быть.
    """
    if not must_contain:
        return None

    def prepare(text: str) -> str:
        return " ".join(text.split()).lower().replace("ё", "е")

    haystack = prepare(answer)
    return all(
        any(prepare(variant) in haystack for variant in needle.split("|") if variant.strip())
        for needle in must_contain
    )


# Язык корпуса. Одно место, потому что от него зависит поведение метрики:
# «ответ не на том языке» определяется относительно этой буквы, а не
# относительно интуиции читателя таблицы.
CORPUS_LANGUAGE = "ru"

# Доля букв одного алфавита, начиная с которой язык считается определённым.
#
# Числа подобраны не из красоты. Русский технический ответ законно содержит
# латиницу: имена переменных (`TG_SNAPSHOT_CACHE`), коды ошибок (`E-1042`),
# названия сервисов. На наших ответах доля латиницы доходит до 45 % при
# совершенно русском тексте — поэтому порог кириллицы низкий.
#
# Иероглифы — наоборот: в русском техническом тексте их не бывает вообще,
# поэтому им достаточно заметной примеси, чтобы объявить язык.
CYRILLIC_FLOOR = 0.35
HAN_CEILING = 0.15


def answer_language(answer: str) -> str:
    """Самая дешёвая проверка из всех: на каком языке написан ответ.

    Зачем она понадобилась. Образец на LlamaIndex ответил не по-русски на 22
    вопросах из 113 — в основном по-китайски: встроенный промпт фреймворка
    английский и языка не задаёт, а `qwen2.5` — китайская модель, и на
    коротких вопросах да/нет она сваливается в родной язык.

    Ни один наш прибор этого не увидел. Схема бы не заметила — текст как
    текст. Цитаты не заметили бы. Судья до этих строк не дошёл. Единственное,
    что сработало, — `answer_contains`, и сработало ОБМАНЧИВО: подстроки у нас
    русские, ответ на китайском не может их содержать, даже если он по сути
    верный. То есть метрика упала по правильной причине и показала при этом
    не ту болезнь: «система не нашла факт» вместо «система ответила на другом
    языке». Лечатся они совершенно по-разному.

    Отсюда правило, которое эта функция закрывает: **метрика, способная
    падать по двум разным причинам, обязана иметь рядом вторую, которая эти
    причины различает.**

    Считается по буквам, без модели и без словарей, из сохранённого текста.
    Поэтому её можно досчитать по старым прогонам (`eval.py recheck`) — а без
    этого новая цифра появилась бы только в будущих прогонах и сравнивать
    `alt1` с `wide3` по ней было бы нельзя.

    Возвращает `"ru"`, `"zh"`, `"en"` или `""` — последнее значит «букв нет,
    определять нечего»: пустой ответ или одни цифры. Пустая строка здесь —
    это прочерк, а не «язык неизвестен и потому плохой».
    """
    letters = [character for character in answer if character.isalpha()]
    if not letters:
        return ""

    cyrillic = sum(1 for character in letters if "Ѐ" <= character <= "ӿ")
    han = sum(1 for character in letters if "一" <= character <= "鿿")
    total = len(letters)

    # Иероглифы проверяются ПЕРВЫМИ и по низкому порогу.
    #
    # Порядок не косметический. Китайский ответ про наш корпус почти всегда
    # содержит латинские имена сервисов (`PIPEPASS`, `FATIGUE-API`), и по
    # доле латиницы его легко принять за английский. А разница существенная:
    # английский ответ читатель вики хотя бы прочтёт.
    if han / total > HAN_CEILING:
        return "zh"
    if cyrillic / total >= CYRILLIC_FLOOR:
        return "ru"
    return "en"


def separation_margin(cosines: list[float], *, tail_from: int = 1, tail_size: int = 5) -> float:
    """Насколько первый результат выделяется на фоне следующих.

    Гипотеза, которую эта величина проверяет: у вопроса, ответ на который в
    корпусе есть, лучший фрагмент должен стоять ЗАМЕТНО выше остальных, а у
    вопроса без ответа все результаты похоже-посредственные. Если это так,
    разрыв разделяет мусор и пользу лучше, чем абсолютный косинус — у которого
    на bge-m3 распределения отвечаемых и неотвечаемых вопросов перекрываются.

    Считается как разность между первым косинусом и средним следующих.
    """
    if len(cosines) < 2:
        return 0.0
    tail = cosines[tail_from : tail_from + tail_size]
    if not tail:
        return 0.0
    return cosines[0] - statistics.fmean(tail)


def _round(value: float | None) -> float | None:
    """Округление, переживающее «мерить нечего». None остаётся None."""
    return None if value is None else round(value, 4)


@dataclass(slots=True)
class RetrievalScore:
    """Метрики поиска по одному вопросу."""

    # None означает «мерить нечего», а НЕ «ноль».
    #
    # У вопроса про живые данные ожидаемых документов нет: «дай топ 5 труб»
    # не написано ни в одном документе корпуса, ответ приходит из сервиса.
    # Пока здесь стоял ноль, двенадцать таких вопросов утянули общий
    # recall@5 с 0.977 до 0.928 — метрика поиска отчиталась о провале на
    # вопросах, где поиска не было вовсе.
    #
    # Это та же ошибка, что мы уже разбирали с judge_ok и citations_ok, и
    # она повторяется потому, что ноль выглядит как нормальное число.
    recall_at_5: float | None
    recall_at_10: float | None
    precision_at_5: float | None
    mrr: float | None
    ndcg_at_10: float | None
    context_hit: bool          # ожидаемая подстрока реально попала в контекст
    best_cosine: float
    passed_floor: bool
    chunk_rank: int = 0        # позиция нужного чанка в выдаче, 0 — не найден
    chunk_mrr: float | None = 0.0  # обратный ранг нужного чанка; None — мерить нечего
    margin: float = 0.0        # разрыв между первым результатом и следующими

    def as_dict(self) -> dict:
        return {
            "recall@5": _round(self.recall_at_5),
            "recall@10": _round(self.recall_at_10),
            "precision@5": _round(self.precision_at_5),
            "mrr": _round(self.mrr),
            "ndcg@10": _round(self.ndcg_at_10),
            "context_hit": self.context_hit,
            "best_cosine": round(self.best_cosine, 4),
            "passed_floor": self.passed_floor,
            "chunk_rank": self.chunk_rank,
            "chunk_mrr": _round(self.chunk_mrr),
            "margin": round(self.margin, 4),
        }


def score_retrieval(
    *,
    retrieved_docs: list[str],
    relevant_docs: set[str],
    context_text: str,
    must_contain: list[str],
    best_cosine: float,
    passed_floor: bool,
    chunk_texts: list[str] | None = None,
    cosines: list[float] | None = None,
) -> RetrievalScore:
    """Считает метрики поиска по одному вопросу.

    `context_hit` — самая практичная из них: ожидаемая подстрока не просто
    лежит в найденном документе, а реально доехала до контекста модели. Между
    «документ нашёлся» и «нужный абзац попал в промпт» помещаются порог
    отсечения, бюджет токенов и дедупликация — то есть три места, где нужный
    текст ещё может отвалиться.
    """
    normalized = " ".join(context_text.split()).lower()
    hit = all(" ".join(needle.split()).lower() in normalized for needle in must_contain)

    rank = chunk_rank(chunk_texts or [], must_contain)

    # Нечего искать — нечего и мерить. Ноль здесь означал бы «искали и не
    # нашли», а мы не искали: у вопроса нет ожидаемых документов.
    nothing_to_find = not relevant_docs

    return RetrievalScore(
        recall_at_5=None if nothing_to_find else recall_at_k(retrieved_docs, relevant_docs, 5),
        recall_at_10=None if nothing_to_find else recall_at_k(retrieved_docs, relevant_docs, 10),
        precision_at_5=(
            None if nothing_to_find else precision_at_k(retrieved_docs, relevant_docs, 5)
        ),
        mrr=None if nothing_to_find else mrr(retrieved_docs, relevant_docs),
        ndcg_at_10=None if nothing_to_find else ndcg_at_k(retrieved_docs, relevant_docs, 10),
        context_hit=hit if must_contain else True,
        best_cosine=best_cosine,
        passed_floor=passed_floor,
        chunk_rank=rank,
        # Нечего искать в чанках — нечего и мерить. Без `must_contain`
        # «нужный чанк не найден» неотличимо от «нужного чанка не было».
        chunk_mrr=(None if not must_contain else (1.0 / rank if rank else 0.0)),
        margin=separation_margin([value for value in (cosines or []) if value is not None]),
    )


def cohen_kappa(first: list[bool], second: list[bool]) -> float:
    """Каппа Коэна: согласие двух разметчиков СВЕРХ случайного.

    Зачем она нужна вместо простого процента совпадений. Если 90 % ответов
    правильные, то разметчик, штампующий «правильно» не глядя, совпадёт с
    человеком в 90 % случаев — и простой процент покажет отличное согласие
    при нулевой полезности. Каппа вычитает из согласия то, что объясняется
    случаем: ноль — «как монетка», единица — полное совпадение.

    Ниже 0.6 судью считаем шумом, и все метрики, построенные на нём, — тоже.
    """
    if len(first) != len(second) or not first:
        raise ValueError("разметки разной длины или пустые")

    total = len(first)
    agree = sum(1 for a, b in zip(first, second) if a == b) / total

    p_first = sum(first) / total
    p_second = sum(second) / total
    chance = p_first * p_second + (1 - p_first) * (1 - p_second)

    if chance >= 1.0:
        # Обе разметки поставили одну и ту же метку везде: согласие
        # объясняется случаем целиком, и каппа не определена.
        return 0.0
    return (agree - chance) / (1 - chance)


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def spread(values: list[float]) -> float:
    """Разброс между прогонами — он же порог значимости.

    Изменение метрики меньше этого числа не является улучшением, каким бы
    приятным оно ни выглядело.
    """
    return round(max(values) - min(values), 4) if len(values) > 1 else 0.0
