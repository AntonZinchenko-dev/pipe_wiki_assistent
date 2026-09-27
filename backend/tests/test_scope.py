"""Вопрос не про вики — отвечать нечем, и признаётся это без модели.

Из живой переписки и прогонов система уверенно отвечала на реплики, в
которых нет вопроса:

    «как дела?»  -> «Текущий статус: сервис работает, табло заполнились...»
    «что»        -> «Труба не списывается по расчёту усталости...»
    «чо пао чем» -> развёрнутый рассказ про постмортемы

Это хуже неверного ответа: неверный человек может заметить, а этот выглядит
как работа.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.scope import nothing_to_search, stems, vocabulary

CORPUS = Path(__file__).resolve().parent.parent.parent / "corpus" / "source"
GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden.jsonl"


def corpus_vocabulary() -> set[str]:
    texts = [path.read_text(encoding="utf-8") for path in CORPUS.glob("*.md")]
    if not texts:
        pytest.skip("корпус недоступен в этом окружении")
    return vocabulary(texts)


@pytest.mark.parametrize("junk", ["как дела?", "что", "чо пао чем", "сосал?"])
def test_a_reply_without_a_question_is_turned_away(junk: str) -> None:
    assert nothing_to_search(junk, corpus_vocabulary()) is True


@pytest.mark.parametrize(
    "question",
    [
        "когда труба списывается",
        "какой порог по усталости считается аварийным",
        "что делать если отчёт долго собирается",
        # Идентификатор — ссылка на наш предмет сам по себе. Такой вопрос не
        # отсекается никогда, даже если остальные слова мимо словаря.
        "что с трубой PP-0007",
    ],
)
def test_a_real_question_is_never_turned_away(question: str) -> None:
    assert nothing_to_search(question, corpus_vocabulary()) is False


def test_the_whole_golden_set_is_clean() -> None:
    """Ноль ложных срабатываний на всех 134 вопросах.

    Отказать на честном вопросе дороже, чем ответить на мусорный: первое
    человек запомнит как «эта штука не работает». Поэтому проверка на весь
    набор, а не на горсть примеров.
    """
    corpus = corpus_vocabulary()
    rows = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]

    wrongly = [
        (r["id"], r["question"])
        for r in rows
        if r.get("answerable") and nothing_to_search(r["question"], corpus)
    ]
    assert wrongly == [], f"отсечены честные вопросы: {wrongly}"


def test_an_empty_index_disables_the_check() -> None:
    """На пустом индексе проверка молчит.

    Отвечать «это не про вики», когда индекс просто не построен, значит
    валить свою недонастройку на пользователя.
    """
    assert nothing_to_search("как дела?", set()) is False


def test_stems_glue_the_cases_together() -> None:
    """Обрезка по пяти буквам склеивает падежи — больше от неё и не нужно."""
    assert stems("трубы трубе трубой") == {"трубы", "трубе", "трубо"}
    assert stems("как") == set(), "слова короче четырёх букв не в счёт"
