"""Тесты метрик. Чистая математика, поэтому проверяется без модели и индекса.

Смысл этих тестов не в том, что формулы «работают», а в том, что каждая
метрика измеряет именно то свойство, ради которого её взяли. Метрика,
считающая что-то не то, хуже отсутствия метрики: на неё будут ориентироваться.
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    cohen_kappa, mrr, ndcg_at_k, precision_at_k, recall_at_k, score_retrieval, spread,
)


def test_recall_finds_relevant_within_k() -> None:
    assert recall_at_k(["A", "B", "C"], {"C"}, 5) == 1.0
    assert recall_at_k(["A", "B", "C"], {"C"}, 2) == 0.0
    assert recall_at_k(["A", "B"], {"A", "B"}, 5) == 1.0
    assert recall_at_k(["A", "X"], {"A", "B"}, 5) == 0.5


def test_recall_without_relevant_documents_is_zero_not_error() -> None:
    """Для неотвечаемого вопроса релевантных документов нет.

    Метрика обязана вернуть ноль, а не упасть: такие вопросы в наборе есть по
    требованию чек-листа, и прогон не должен на них ломаться.
    """
    assert recall_at_k(["A"], set(), 5) == 0.0


def test_mrr_rewards_higher_position() -> None:
    """MRR отвечает на вопрос «насколько высоко правильный ответ».

    Для RAG это важнее precision: фрагмент с первой позиции модель использует
    охотнее, чем с восьмой.
    """
    assert mrr(["A", "B", "C"], {"A"}) == 1.0
    assert mrr(["A", "B", "C"], {"B"}) == 0.5
    assert mrr(["A", "B", "C"], {"C"}) == pytest.approx(1 / 3)
    assert mrr(["A", "B"], {"Z"}) == 0.0


def test_precision_at_k_counts_only_top() -> None:
    assert precision_at_k(["A", "B", "C", "D", "E"], {"A", "B"}, 5) == pytest.approx(0.4)
    assert precision_at_k([], {"A"}, 5) == 0.0


def test_ndcg_sees_order_while_recall_does_not() -> None:
    """Ключевая проверка: именно поэтому в наборе метрик есть nDCG.

    Два результата с ОДИНАКОВЫМ составом, но разным порядком. recall не
    отличит их вообще — а значит не заметит и работу реранкера, который
    только порядок и меняет.
    """
    good = ["gold", "x", "y", "z"]
    bad = ["x", "y", "z", "gold"]
    relevant = {"gold"}

    assert recall_at_k(good, relevant, 10) == recall_at_k(bad, relevant, 10)
    assert ndcg_at_k(good, relevant, 10) > ndcg_at_k(bad, relevant, 10)
    assert ndcg_at_k(good, relevant, 10) == pytest.approx(1.0)


def test_ndcg_is_one_when_all_relevant_are_first() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 10) == pytest.approx(1.0)


# ------------------------------------------------------------- контекст-хит


def test_context_hit_requires_substring_in_actual_context() -> None:
    """«Документ нашёлся» и «нужный абзац доехал до модели» — разные события.

    Между ними стоят порог отсечения, бюджет токенов и дедупликация: три
    места, где нужный текст ещё может отвалиться.
    """
    score = score_retrieval(
        retrieved_docs=["TG-GW"],
        relevant_docs={"TG-GW"},
        context_text="Кэш включается переменной TG_SNAPSHOT_CACHE=on и требует перезапуска",
        must_contain=["TG_SNAPSHOT_CACHE=on"],
        best_cosine=0.7,
        passed_floor=True,
    )
    assert score.recall_at_5 == 1.0
    assert score.context_hit


def test_context_hit_false_when_document_found_but_fragment_lost() -> None:
    score = score_retrieval(
        retrieved_docs=["TG-GW"],
        relevant_docs={"TG-GW"},
        context_text="Шлюз принимает поток телеметрии и приводит теги к словарю",
        must_contain=["TG_SNAPSHOT_CACHE=off"],
        best_cosine=0.7,
        passed_floor=True,
    )
    assert score.recall_at_5 == 1.0, "документ нашёлся"
    assert not score.context_hit, "но нужный фрагмент до контекста не дошёл"


def test_context_hit_tolerates_whitespace() -> None:
    score = score_retrieval(
        retrieved_docs=["RL-92"],
        relevant_docs={"RL-92"},
        context_text="Пороги:\n  внимание   50 %,\n  авария 80 %",
        must_contain=["внимание 50 %"],
        best_cosine=0.6,
        passed_floor=True,
    )
    assert score.context_hit


# -------------------------------------------------------------------- каппа


def test_kappa_is_one_on_full_agreement() -> None:
    judge = [True, False, True, False, True, False]
    assert cohen_kappa(judge, judge) == pytest.approx(1.0)


def test_kappa_near_zero_when_judge_says_yes_to_everything() -> None:
    """Главная причина, по которой берут каппу, а не процент совпадений.

    Судья, штампующий «верно» не глядя, совпадёт с человеком в 90 % случаев
    на наборе, где 90 % ответов верные, — и процент покажет прекрасное
    согласие при нулевой полезности. Каппа это вычитает.
    """
    human = [True] * 9 + [False]
    lazy_judge = [True] * 10

    agreement = sum(1 for j, h in zip(lazy_judge, human) if j == h) / len(human)
    assert agreement == pytest.approx(0.9)
    assert cohen_kappa(lazy_judge, human) == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_negative_when_judge_disagrees_systematically() -> None:
    human = [True, True, False, False]
    inverted = [False, False, True, True]
    assert cohen_kappa(inverted, human) < 0


def test_kappa_rejects_mismatched_input() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


# ------------------------------------------------------------------- разброс


def test_spread_is_the_significance_threshold() -> None:
    """Разброс между прогонами без изменений — то, ниже чего «улучшение» шум."""
    assert spread([0.81, 0.83, 0.80]) == pytest.approx(0.03)
    assert spread([0.81]) == 0.0


# ------------------------------------ метрики на уровне чанков и разделимость


def test_chunk_rank_finds_position_of_required_text() -> None:
    """Позиция нужного ЧАНКА — не насыщающаяся метрика.

    На первом настоящем прогоне recall по документам оказался 1.000 по всем
    типам вопросов: документов шестнадцать, в выдачу попадает четверть
    индекса. Насыщенная метрика не измеряет ничего — она не покажет ни
    улучшения, ни деградации. Позиция чанка различает «стоял первым» и
    «стоял восьмым», а это ровно то, что меняет реранкер.
    """
    from eval.metrics import chunk_rank

    chunks = [
        "Шлюз принимает поток телеметрии с буровых установок",
        "Кэш включается переменной TG_SNAPSHOT_CACHE=on, нужен перезапуск",
        "Задержка доставки не более 800 мс",
    ]
    assert chunk_rank(chunks, ["TG_SNAPSHOT_CACHE=on"]) == 2
    assert chunk_rank(chunks, ["800 мс"]) == 3
    assert chunk_rank(chunks, ["чего тут нет"]) == 0
    assert chunk_rank(chunks, []) == 0


def test_chunk_rank_requires_all_needles_in_one_chunk() -> None:
    from eval.metrics import chunk_rank

    chunks = ["первое условие", "второе условие", "первое и второе условие вместе"]
    assert chunk_rank(chunks, ["первое", "второе"]) == 3


def test_margin_measures_how_much_the_top_result_stands_out() -> None:
    from eval.metrics import separation_margin

    standout = separation_margin([0.80, 0.50, 0.48, 0.47])
    flat = separation_margin([0.55, 0.54, 0.53, 0.52])
    assert standout > flat
    assert separation_margin([0.7]) == 0.0
    assert separation_margin([]) == 0.0


def test_separability_reports_upper_bound_of_a_feature() -> None:
    """Разделимость — это ВЕРХНЯЯ граница возможностей признака.

    Если даже лучший порог даёт плохой баланс, признак не годится, и крутить
    порог бессмысленно. Ровно этот вывод и получился на живых данных для
    абсолютного косинуса.
    """
    from eval.diagnose import separability

    clean = separability([0.8, 0.85, 0.9], [0.2, 0.25, 0.3], feature="cosine")
    assert clean.balanced == pytest.approx(1.0)
    assert clean.overlap == 0.0

    overlapping = separability([0.5, 0.55, 0.6], [0.45, 0.52, 0.58], feature="cosine")
    assert overlapping.balanced < 1.0
    assert overlapping.overlap > 0.0


def test_separability_handles_empty_groups() -> None:
    from eval.diagnose import separability

    result = separability([], [0.3], feature="cosine")
    assert result.balanced == 0.0


def test_nothing_to_find_is_not_a_failed_search() -> None:
    """Вопрос без ожидаемых документов не может иметь recall.

    Из живого прогона: двенадцать вопросов про живые данные утянули общий
    recall@5 с 0.977 до 0.928. Метрика поиска отчиталась о провале там, где
    поиска не было вовсе: «дай топ 5 труб» не написано ни в одном документе
    корпуса, ответ приходит из сервиса.

    Ноль выглядит как нормальное число — потому эта ошибка и повторяется.
    """
    from eval.metrics import score_retrieval

    score = score_retrieval(
        retrieved_docs=["FA-API", "PP-OVW"],
        relevant_docs=set(),
        context_text="Таблица T98D4: топ-5 труб.",
        must_contain=[],
        best_cosine=0.2,
        passed_floor=False,
    )

    assert score.recall_at_5 is None
    assert score.mrr is None
    assert score.ndcg_at_10 is None
    assert score.chunk_mrr is None
    # И в сериализации None остаётся None, а не превращается в ноль.
    assert score.as_dict()["recall@5"] is None


def test_a_normal_question_still_gets_its_numbers() -> None:
    """Исключение не должно съесть обычный случай."""
    from eval.metrics import score_retrieval

    score = score_retrieval(
        retrieved_docs=["TG-GW", "PP-OVW"],
        relevant_docs={"TG-GW"},
        context_text="Кэш отключается переменной TG_SNAPSHOT_CACHE=off.",
        must_contain=["TG_SNAPSHOT_CACHE=off"],
        best_cosine=0.7,
        passed_floor=True,
    )

    assert score.recall_at_5 == 1.0
    assert score.context_hit is True
