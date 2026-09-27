"""Таблица обязана сама говорить, что такое одна её строка.

ИСТОРИЯ. На «на какой скважине трубы хуже всего» сервис отдал верную таблицу
скважин, а ответ вышел про трубу — с обозначением, которого нет ни в одном
документе корпуса (я проверил все шестнадцать: обозначений труб там нет
вовсе, то есть имя было сгенерировано в нашем формате).

ПРИЧИНА БЫЛА НАША. Вот что видела модель:

    Топ-5 скважин по максимальной выработке ТРУБ
    Колонки: Скважина, Месторождение, ТРУБ, Макс. выработка

«Труб» — дважды, «Скважина» — один раз. «Макс. выработка» без уточнения, а во
всём корпусе выработка бывает только у трубы. Мы сами сказали, что таблица
про трубы, не меньше раз, чем что она про скважины.

ПОЧЕМУ ЭТО НЕ СЕДЬМАЯ ПОПЫТКА УГОВОРИТЬ МОДЕЛЬ. Мы не добавляем правило «не
путай скважину с трубой». Мы исправляем подпись, которая врала, и сообщаем
факт, которого в описании таблицы не было: что есть одна запись. Единственная
удача за два дня из семи попыток была именно такой — модели дали недостающий
факт, а не запрет.
"""

from __future__ import annotations

from app.rag.result import Column, Dataset


def _wells(rows: list[dict]) -> Dataset:
    from app.rag.live import WELL_COLUMNS

    return Dataset(
        kind="wells.fleet",
        title="Скважины: 5 самых тяжёлых по максимуму выработки",
        columns=list(WELL_COLUMNS),
        rows=rows,
        total_found=22,
        scanned=22,
    )


ROW = {"well_id": "W-122", "field": "Приобское", "pipe_count": 1,
       "max_damage_percent": 97.7}


def test_the_wells_table_no_longer_calls_itself_a_pipe_table() -> None:
    """Слово «труб» больше не стоит там, где читается как предмет строки."""
    from app.rag.live import WELL_COLUMNS, _wells_title

    title = _wells_title("", 5, 0)
    assert "скважин" in title.lower()
    # НИ ОДНОГО «труб» в заголовке. Проверка была слабее — «не заканчивается
    # на труб» — и её прошёл заголовок «...по своей худшей трубе», после
    # которого модель написала «самая изношенная труба парка — скважина
    # W-122». Слабая проверка хуже отсутствующей: она создаёт уверенность.
    assert "труб" not in title.lower(), title

    titles = [column.title for column in WELL_COLUMNS]
    assert titles[0] == "Скважина", "предмет строки обязан стоять первым"
    # «Труб» среди колонок РОВНО ОДИН раз, и только как счётчик: спутать
    # счётчик с предметом строки нельзя, а «Выработка худшей ТРУБЫ» — можно.
    with_pipe = [name for name in titles if "труб" in name.lower()]
    assert with_pipe == ["Труб в скважине"], with_pipe
    # Ни одного заголовка с запятой внутри: строка собирается через «, »,
    # и запятая в заголовке превращает одно поле в два.
    assert not any("," in name for name in titles)


def test_the_digest_says_what_one_row_is() -> None:
    """Это информация, а не просьба: в описании таблицы её просто не было."""
    digest = _wells([ROW]).digest()

    assert "Одна строка — одна СКВАЖИНА" in digest, (
        "выжимка не говорит, что такое одна запись — по колонкам это не восстановить"
    )
    assert "W-122" in digest


def test_every_live_table_says_it() -> None:
    """Пробел закрыт у всех таблиц, а не только у той, на которой поймали.

    Один вид таблицы, забытый здесь, — это тот же дефект, ждущий своего
    вопроса. Дешевле закрыть все четыре сразу.
    """
    from app.rag.result import ROW_MEANS

    assert set(ROW_MEANS) == {
        "wells.fleet", "pipes.fleet", "pipes.inspections", "pipe.passport",
    }


def test_the_leading_row_is_rendered_from_the_table() -> None:
    """Откат собирается из таблицы, а не сочиняется.

    Каждое слово — заголовок колонки и значение из строки. Ни одного слова от
    модели, поэтому выдумке взяться неоткуда.
    """
    lead = _wells([ROW]).leading_row()

    assert "Скважина W-122" in lead
    assert "Приобское" in lead
    assert "97.7" in lead
    assert "PP-" not in lead, "в откате не должно быть обозначений труб вовсе"


def test_an_empty_table_has_no_leading_row() -> None:
    """Пустая таблица — не ответ. Тогда откат остаётся прежним.

    Важно для отказов сервиса: у таблицы с E-1042 строк нет, и «верхняя
    строка» там была бы пустой фразой.
    """
    assert _wells([]).leading_row() == ""


def test_the_fallback_replaces_the_useless_sentence() -> None:
    """Собираем всё вместе: выдумка выброшена, на её месте факт из таблицы."""
    from app.rag.table_answer import drop_invented

    lead = _wells([ROW]).leading_row()
    fixed, dropped = drop_invented(
        "Самая изношенная труба парка — PP-0035, выработано 97.7 % ресурса [1].",
        invented=["PP-0035"],
        tables=1,
        enough=f"Верхняя строка таблицы: {lead}.",
    )

    assert dropped == ["PP-0035"]
    assert "W-122" in fixed
    assert "PP-0035" not in fixed
    assert "Данные в таблице выше" not in fixed


def test_without_the_rule_the_safe_fallback_stays() -> None:
    """Списочный вопрос откатывается по-прежнему, и это не полумера.

    На «дай топ 5» верхняя строка не ответ. «Данные в таблице выше» там
    беднее, но верно, а одна строка вместо пяти была бы новым дефектом.
    """
    from app.rag.table_answer import ENOUGH, drop_invented

    fixed, dropped = drop_invented(
        "Хуже всех PP-0035.", invented=["PP-0035"], tables=1
    )

    assert dropped == ["PP-0035"]
    assert fixed == ENOUGH
