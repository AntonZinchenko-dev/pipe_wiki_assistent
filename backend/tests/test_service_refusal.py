"""Отказ сервиса обязан дойти до человека словами — и это надо мерить.

ЧТО СЛУЧИЛОСЬ. На стенде труба PP-0007 отвечает E-1042 ВСЕГДА: так задумано,
обработку отказов проверяют на отказах. В прогоне на 144 вопросах вопрос «что
с трубой PP-0007» показал все агентские метрики в порядке — и ответ, в котором
про отказ сервиса нет ни слова: вместо него человеку рассказали про архивный
постмортем.

ПОЧЕМУ НИЧТО НЕ СРАБОТАЛО. Требование стояло ЗАМЕТКОЙ в самом вопросе:
«сервис отвечает E-1042; отказ обязан дойти до человека словами». Заметку
читают люди, а метрика её не читает. Всё, что проверялось, — сколько таблиц
пришло; таблица с отказом внутри считается таблицей, и счёт сошёлся.

ОТДЕЛЬНОЕ ПОЛЕ, А НЕ answer_must_contain. На живых вопросах разметка ответа
подстрокой запрещена намеренно: она награждает пересказ строк таблицы
словами. Случай отказа — обратный: таблицы нет и быть не может, и словами
отвечать НАДО. Два обратных требования на одном поле означали бы, что поле
не значит ничего.
"""

from __future__ import annotations

import json

import pytest

from eval.dataset import load


def _write(tmp_path, payload: dict):
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_only_a_service_can_refuse(tmp_path) -> None:
    """Код отказа у вопроса по документам — разметка про несуществующее.

    Отказать может тот, кого зовут. Документ не отказывает: он либо есть в
    индексе, либо нет, и это другая метрика.
    """
    path = _write(tmp_path, {
        "id": "x90", "question": "что означает E-1042", "type": "fact",
        "difficulty": "easy", "answerable": True, "docs": ["FA-API"],
        "must_contain": ["Расчёт для трубы устарел"],
        "answer_must_contain": ["устарел"],
        "expect_error": "E-1042",
    })
    with pytest.raises(ValueError, match="отказывать может только сервис"):
        load(path)


def test_a_refusal_needs_a_call_to_refuse(tmp_path) -> None:
    """Ждём код отказа — значит ждём и вызова, на котором его получат.

    Без этого правила «таблицы нет» нельзя отличить от «сервис отказал»:
    ровно так a006 и выглядел в прогоне — инструмент назван верно, аргумент
    потерян по дороге, до сервиса дело не дошло вовсе.
    """
    path = _write(tmp_path, {
        "id": "x91", "question": "что с трубой PP-0007", "type": "live",
        "difficulty": "medium", "answerable": True, "docs": ["FA-API"],
        "must_contain": [], "answer_must_contain": [],
        "expect_tables": 1, "expect_tools": [],
        "expect_error": "E-1042",
    })
    with pytest.raises(ValueError, match="отказ не на чем получить"):
        load(path)


def test_the_marked_up_case_loads(tmp_path) -> None:
    """Правильная разметка проходит: поле бесполезно, если им нельзя пользоваться."""
    path = _write(tmp_path, {
        "id": "x92", "question": "что с трубой PP-0007", "type": "live",
        "difficulty": "medium", "answerable": True, "docs": ["FA-API"],
        "must_contain": [], "answer_must_contain": [],
        "expect_tables": 1, "expect_tools": ["live_pipe"],
        "expect_error": "E-1042",
    })
    questions = load(path)
    assert questions[0].expect_error == "E-1042"


def test_the_real_set_marks_the_rigged_pipe() -> None:
    """В самом наборе вопрос про PP-0007 размечен кодом отказа.

    Проверка на регресс разметки: без неё вопрос тихо вернётся к состоянию
    «ждём таблицу», в котором пустая таблица с отказом внутри читается как
    успех.
    """
    from pathlib import Path

    for line in Path("eval/golden.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("id") == "a006":
            assert payload.get("expect_error") == "E-1042"
            assert "live_pipe" in payload.get("expect_tools", [])
            return
    raise AssertionError("вопрос a006 в наборе не найден")


def test_the_code_must_be_in_the_answer_and_in_the_table() -> None:
    """Код в ответе — И код в таблице. Одного ответа недостаточно.

    Таблица кодов ошибок лежит в документе, и модель вправе назвать E-1042,
    не сходив в сервис вовсе, — просто прочитав про него в вики. Такой ответ
    выглядит правильным и правильным не является: он не знает, что сервис
    отказал СЕЙЧАС. Поэтому проверяются оба места.
    """
    from eval.runner import _slim

    refused = _slim({
        "title": "Паспорт трубы PP-0007",
        "columns": [], "rows": [], "total_found": 0,
        "error": {"code": "E-1042", "message": "Расчёт устарел", "retriable": True},
    })
    assert refused["error"], "отказ выпал из таблицы прогона — отличать его стало нечем"
    assert "e-1042" in json.dumps(refused, ensure_ascii=False).lower()

    # Пустая таблица без отказа — другой диагноз, и путать их нельзя.
    empty = _slim({"title": "Трубы парка", "columns": [], "rows": [], "total_found": 0})
    assert empty["error"] is None
