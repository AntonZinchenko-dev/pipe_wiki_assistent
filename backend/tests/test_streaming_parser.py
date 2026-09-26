"""Тесты потокового разбора структурированного ответа.

Главный тест здесь — тот, который режет вход на ОДИН символ. Локально куски
приходят целыми, и баг с разорванной границей события физически не может
воспроизвестись на нормальных тестовых данных: он ждёт прода. Поэтому режем
искусственно мелко (раздел 5 чек-листа).
"""

from __future__ import annotations

import json

import pytest

from app.rag.answer import StreamingAnswerParser


def feed_in_pieces(payload: str, size: int) -> tuple[str, StreamingAnswerParser]:
    parser = StreamingAnswerParser()
    out = []
    for start in range(0, len(payload), size):
        out.append(parser.feed(payload[start : start + size]))
    return "".join(out), parser


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13, 64, 4096])
def test_extracts_answer_at_any_chunk_size(size: int) -> None:
    payload = json.dumps(
        {
            "status": "answered",
            "answer": "Кэш включается переменной TG_SNAPSHOT_CACHE=on, "
                      "после этого нужен перезапуск процесса.",
            "citations": [{"fragment": 1, "quote": "требуется перезапуск процесса"}],
        },
        ensure_ascii=False,
    )
    text, parser = feed_in_pieces(payload, size)
    assert parser.finished
    assert text == json.loads(payload)["answer"]


@pytest.mark.parametrize("size", [1, 2, 3, 6])
def test_survives_unicode_escapes_split_across_chunks(size: int) -> None:
    """`\\u0410` разрезанное пополам — ровно тот случай, на котором ломается
    разбор «по куску за раз»."""
    payload = json.dumps(
        {"status": "answered", "answer": "Ответ: Ага\nи ещё «кавычки»", "citations": []},
        ensure_ascii=True,  # заставляем экранировать кириллицу в \uXXXX
    )
    text, parser = feed_in_pieces(payload, size)
    assert parser.finished
    assert text == "Ответ: Ага\nи ещё «кавычки»"


def test_handles_escaped_quotes_inside_answer() -> None:
    payload = json.dumps(
        {"status": "answered", "answer": 'Поле "answer" может содержать кавычки', "citations": []},
        ensure_ascii=False,
    )
    text, parser = feed_in_pieces(payload, 1)
    assert parser.finished
    assert text == 'Поле "answer" может содержать кавычки'


def test_ignores_keys_before_answer() -> None:
    payload = '{"status":"not_found","citations":[],"answer":"нет данных"}'
    text, parser = feed_in_pieces(payload, 1)
    assert text == "нет данных"
    assert parser.finished


def test_does_not_finish_on_incomplete_stream() -> None:
    parser = StreamingAnswerParser()
    delta = parser.feed('{"status":"answered","answer":"начало')
    assert delta == "начало"
    assert not parser.finished


def test_raw_is_preserved_for_final_validation() -> None:
    payload = '{"status":"answered","answer":"текст","citations":[]}'
    _, parser = feed_in_pieces(payload, 3)
    assert json.loads(parser.raw)["status"] == "answered"
