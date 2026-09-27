"""Идеальные цифры при плохих ответах — то, что эта метрика делает видимым.

ЧТО СЛУЧИЛОСЬ. Прогон, в котором ВСЁ агентское показало ровно 1.000:

    tables_ok 1.000   tools_ok 1.000   live_clean 1.000   table_answered 1.000

И два живых ответа из девяти, в которых таблицы как будто не было:

    «какие трубы под списание»   -> ответ целиком из регламента,
                                    пришедшую таблицу не упомянул вовсе
    «сколько труб в парке»       -> «в парке 400 труб» из документа,
                                    хотя сервис отдал выборку

Всё, что мерили метрики, — ДОСТАВЛЕНА ли таблица и не ВЫДУМАНО ли лишнее.
Отвечает ли текст по этой таблице, не мерило ничто.

ПОЧЕМУ БЕЗ РАЗМЕТКИ. У живых вопросов проверка ответа подстрокой запрещена
намеренно — она награждала бы пересказ строк словами. Здесь разметка не нужна:
проверка считается из того, что ПРИШЛО. Значит правило валидатора остаётся
нетронутым, а пятно становится видно.
"""

from __future__ import annotations

from eval.metrics import answer_uses_table

PIPES = {
    "title": "Топ-5 труб по выработке ресурса",
    "columns": [{"key": "pipe_id", "title": "Труба"},
                {"key": "damage_percent", "title": "Выработка"}],
    "rows": [{"pipe_id": "PP-0035", "damage_percent": 97.7},
             {"pipe_id": "PP-0036", "damage_percent": 94.1}],
    "error": None,
}

REFUSED = {
    "title": "Паспорт трубы PP-0007",
    "columns": [{"key": "pipe_id", "title": "Труба"}],
    "rows": [],
    "error": {"code": "E-1042", "message": "Расчёт устарел"},
}


def test_an_answer_from_the_documents_is_caught() -> None:
    """Главный случай: таблица пришла, ответ из регламента."""
    answer = (
        "Труба списывается, если инспектор проставил категорию SCRAP, либо если "
        "обнаружена сквозная трещина или смятие замка [6]."
    )
    assert answer_uses_table(answer, [PIPES]) is False


def test_naming_a_row_passes() -> None:
    """Назвал объект из таблицы — проверка пройдена."""
    answer = "Выработка ресурса у трубы PP-0035 составляет 97.7 % [1]."
    assert answer_uses_table(answer, [PIPES]) is True


def test_our_own_fallback_is_not_judged() -> None:
    """«Данные в таблице выше» — НАША формулировка, а не ответ модели.

    Спрашивать с неё объектов бессмысленно: она для случая, когда добавить к
    таблице нечего, и объектов в ней быть не должно. Без этого исключения
    метрика ругала бы нас за нашу же правильную починку.
    """
    from app.rag.table_answer import ENOUGH

    assert answer_uses_table(ENOUGH, [PIPES]) is None
    assert answer_uses_table(
        "Верхняя строка таблицы: Скважина W-122, Максимум выработки 97.7 %.",
        [PIPES],
    ) is None


def test_a_refused_table_has_nothing_to_name() -> None:
    """У отказа сервиса строк нет — значит и объектов назвать нельзя.

    Считать это провалом значило бы наказывать ответ за то, что сервис лёг.
    За такие ответы отвечает `refusal_said`, и мерит он другое.
    """
    answer = "Запрос «Паспорт трубы PP-0007» не удался. Код E-1042 [1]."
    assert answer_uses_table(answer, [REFUSED]) is None


def test_no_table_means_nothing_to_check() -> None:
    """Вопрос по документам эта метрика не касается вовсе."""
    assert answer_uses_table("Порог внимания — 60 % [1].", []) is None


def test_case_does_not_hide_a_hit() -> None:
    """«pp-0035» в ответе и «PP-0035» в строке — один объект."""
    assert answer_uses_table("Хуже всего pp-0035.", [PIPES]) is True


def test_the_check_is_weak_in_one_direction_and_says_so() -> None:
    """`True` — слабый сигнал, и на это полагаться нельзя.

    Ответ ниже называет объект таблицы и всё равно плохой: он приписал трубе
    код ошибки вместо процента. Метрика его пропустит, и это НЕ дефект
    метрики: она задаёт нижнюю границу, как `answer_contains`, а не заменяет
    судью. Тест стоит здесь, чтобы граница была записана, а не подразумевалась.
    """
    bad = "В таблице есть трубы: PP-0035 (97.7 %) и PP-0036 (E-5001)."
    assert answer_uses_table(bad, [PIPES]) is True


def test_the_run_and_the_report_carry_it() -> None:
    """Метрика, которой нет в сводке, не существует.

    Проверено на своей шкуре: счётчик неудачных ссылок однажды не входил в
    сводку, и правка, чинившая ровно цитаты, показала «починилось 0».
    """
    from pathlib import Path

    from eval.report import SUMMARY_KEYS
    from eval.runner import RowResult

    assert "answer_uses_table" in SUMMARY_KEYS
    assert hasattr(
        RowResult(question_id="x", question="?", type="live", difficulty="easy",
                  answerable=True, critical=False, retrieved_docs=[], retrieval={}),
        "answer_uses_table",
    )
    source = Path("eval/runner.py").read_text(encoding="utf-8")
    assert '"answer_uses_table": _rate(' in source, "метрика не попадает в сводку"
