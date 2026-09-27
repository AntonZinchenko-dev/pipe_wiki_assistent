"""Судья и таблицы: прибор, молчавший там, где система врала.

`judge_ok` в прогоне был 0.963 — по 107 вопросам из 134. Среди исключённых
оказались пять живых вопросов из девяти, с причиной «нет выдержек для
сверки»: судья сличает утверждения с выдержками из ДОКУМЕНТОВ, а для «дай
топ 5 труб» документной выдержки не существует и существовать не может.

То есть прибор молчал ровно там, где ответы разваливались. Это второй такой
случай за день — `answer_contains` тоже освобождает живые вопросы от
проверок и потому РОС, пока ответы по таблицам ломались.
"""

from __future__ import annotations

from eval.judge import Judge

TABLE = {
    "title": "Трубы парка",
    "total_found": 39,
    "columns": [{"key": "pipe_id", "title": "Труба"}, {"key": "damage", "title": "Выработка"}],
    "rows": [{"pipe_id": "PP-0035", "damage": "97.7 %"}, {"pipe_id": "PP-0007", "damage": "91.2 %"}],
}


def test_the_table_becomes_an_excerpt() -> None:
    excerpt = Judge.table_excerpt([TABLE])

    assert "PP-0035" in excerpt
    assert "97.7 %" in excerpt
    assert "Труба" in excerpt and "Выработка" in excerpt


def test_the_excerpt_separates_found_from_shown() -> None:
    """«Найдено 39, показано 2» и «найдено всего 2» — разные факты.

    Именно на этой разнице ответы и врут: «в таблице показаны только три
    трубы» звучит как жалоба на нехватку данных, хотя три строки и были
    всем, что просили. Судья обязан видеть оба числа, иначе он не отличит
    неполный ответ от полного.
    """
    excerpt = Judge.table_excerpt([TABLE])

    assert "39" in excerpt
    assert "строк 2" in excerpt


def test_no_tables_means_no_excerpt() -> None:
    """Пусто — значит пусто. Выдуманная выдержка хуже отсутствующей."""
    assert Judge.table_excerpt([]) == ""
