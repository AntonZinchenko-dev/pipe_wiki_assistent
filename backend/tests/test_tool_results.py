"""Результат инструмента как данные: что именно это чинит.

Переделка сделана не ради красоты контракта. Она убирает ЦЕЛЫЙ КЛАСС
ошибок, а не уменьшает его: модель больше не переписывает таблицу, значит
и переписать её неправильно не может.

Три живых промаха, с которых всё началось, — три разные модели, одна
болезнь:

    PP-0001, выработка 7 %      (в сервисе 77.0)
    PP-0029, выработка 72.0 %   (такой трубы в парке нет)
    «Топ-5: данные для пятой трубы недоступны»  (были все пять)

Здесь проверяется, что новый путь эти случаи исключает по построению, а
не уговорами в промпте.
"""

from __future__ import annotations

from app.rag.result import Column, Dataset, OpenTable, ToolError, open_tables_block

COLUMNS = [
    Column(key="pipe_id", title="Труба"),
    Column(key="damage_percent", title="Выработка", unit="%", kind="number"),
]


def fleet(rows: int, *, total: int | None = None, truncated: bool = False) -> Dataset:
    return Dataset(
        kind="pipes.fleet",
        title="Топ труб по выработке",
        columns=COLUMNS,
        rows=[
            {"pipe_id": f"PP-{number:04d}", "damage_percent": 90.0 - number}
            for number in range(1, rows + 1)
        ],
        total_found=total if total is not None else rows,
        scanned=40,
        truncated=truncated,
        taken_at="25.09.2026 12:00",
    )


def test_two_examples_are_a_pattern_to_continue() -> None:
    """Один пример — это пример. Два примера — образец для подражания.

    Пока в выжимку шли первая и последняя строки, они читались как начало
    и конец последовательности, а последовательность хочется продолжить.
    Середину модель дописывала из головы.
    """
    digest = fleet(20).digest()

    assert "PP-0001" in digest
    assert "PP-0020" not in digest, "последняя строка — приглашение достроить середину"


def test_the_model_never_sees_the_whole_table() -> None:
    """Главное свойство переделки, и проверять его надо первым.

    Если строки просочатся в выжимку, вернётся ровно то, от чего уходили:
    модель начнёт их переписывать.
    """
    data = fleet(20)
    digest = data.digest()

    assert "PP-0001" in digest, "первая строка нужна для связного текста"
    assert "PP-0010" not in digest, "середина таблицы модели не показывается"
    assert digest.count("PP-") <= 3


def test_the_table_goes_to_the_browser_in_full() -> None:
    """Пользователь видит БОЛЬШЕ, чем раньше, а не меньше.

    Прежний предел в двенадцать строк был ограничением КОНТЕКСТА: каждая
    строка стоила токенов. Строк в контексте больше нет, и платить старую
    цену незачем.
    """
    shown = fleet(40).to_dict()

    assert len(shown["rows"]) == 40
    assert shown["columns"][0]["title"] == "Труба"


def test_how_many_were_found_survives() -> None:
    """«Нашлось 40, показано 5» — то, что первым терялось в пересказе.

    Без этого «топ-5» читается как «в парке пять труб», и человек делает
    вывод по выборке, приняв её за весь парк.
    """
    digest = fleet(5, total=40, truncated=True).digest()

    assert "Найдено записей: 40" in digest
    assert "Показано пользователю: 5" in digest
    assert "ещё 35 записей, они не показаны" in digest


def test_the_fragment_holds_facts_and_nothing_else() -> None:
    """Инструкциям во фрагменте не место, и вот чем это кончилось.

    Правила «не пересказывай таблицу» лежали внутри фрагмента, вместе с
    данными. Модель их пересказала пользователю:

        «В таблице T98D4, которая уже показана пользователю на экране
         целиком, отображены топ-5 труб. Остальные строки не показаны…»

    И сослалась на них как на источник — процитировала служебную строку
    как содержание документа. Всё, что лежит во фрагменте, модель считает
    материалом для цитирования: так мы её и учили.
    """
    digest = fleet(5).digest()

    assert "Найдено записей" in digest, "факты остаются"
    for instruction in ("УЖЕ ПОКАЗАНА", "не переписывай", "Не переписывай", "выдумкой"):
        assert instruction not in digest, f"инструкция «{instruction}» просочилась в цитируемое"


def test_the_rules_live_in_the_message_next_to_the_question() -> None:
    """Их место там, где модель ищет указания, а не источники."""
    from app.rag.result import TABLE_RULES

    assert "ТАБЛИЦА И ЕСТЬ ОТВЕТ" in TABLE_RULES
    # Не пересказывать сами правила — модель это делала: «я могу показать
    # только первую строку таблицы» написано человеку поверх полной таблицы.
    assert "наша кухня" in TABLE_RULES


def test_having_nothing_to_add_is_a_valid_answer() -> None:
    """Пустое место в конце ответа модель заполняет выдумкой.

    Из живого прогона: на таблицу скважин, про которые в вики нет ни
    одного правила, модель сочинила сервис WELLD, скважину
    WELL-00123456789 и три даты бурения. Сказать ей «не выдумывай» мало —
    надо дать разрешение промолчать.
    """
    from app.rag.result import TABLE_RULES

    assert "Данные в таблице выше." in TABLE_RULES
    assert "ЕСЛИ ДОБАВИТЬ НЕЧЕГО" in TABLE_RULES
    # И отдельным запретом — совет сходить в другую систему за тем, что
    # человек уже получил.
    assert "обратиться в другую систему" in TABLE_RULES


def test_the_tool_never_invites_anyone_to_the_next_page() -> None:
    """Листает человек, а не агент, — значит и приглашения быть не должно.

    Метка лежала в выжимке для агента, и агент вёл себя ровно так, как его
    попросили: видел её и шёл за следующей страницей. На «дай топ 10 труб»
    он приносил двадцать строк, на «как дела?» — восемнадцать.

    Метка по-прежнему уходит в браузер (поле `next_cursor` таблицы) и
    возвращается оттуда со следующим вопросом, если человек попросил ещё.
    Это единственный путь, которым она попадает к агенту.
    """
    page = fleet(5, total=39, truncated=True)
    page.next_cursor = "fleet|0.0000|5|5"

    assert "cursor" not in page.agent_note()
    assert "cursor" not in page.digest()
    # В браузер метка уходит — иначе листать было бы нечем.
    assert page.to_dict()["next_cursor"] == "fleet|0.0000|5|5"


def test_a_single_row_is_shown_to_the_model_whole() -> None:
    """Паспорт одной трубы — исключение, и оно осмысленное.

    На вопрос «какая выработка у PP-0035» ответом служит само число.
    Спрятать его от модели значило бы запретить ей ответить.
    """
    one = Dataset(
        kind="pipe.passport", title="Паспорт PP-0035", columns=COLUMNS,
        rows=[{"pipe_id": "PP-0035", "damage_percent": 97.7}], total_found=1,
    )

    assert "97.7" in one.digest()


def test_an_error_carries_what_to_do_next_not_just_what_broke() -> None:
    """«Подождите минуту» и «такой трубы нет» — разные ответы человеку.

    До этого оба выглядели как «произошла ошибка», и человек шёл пробовать
    снова там, где пробовать бессмысленно.
    """
    stale = Dataset(
        kind="pipe.passport", title="Паспорт PP-0007",
        error=ToolError(
            code="E-1042", message="Расчёт устарел.",
            retriable=True, retry_after_s=60, hint="Значение появится после пересчёта.",
        ),
    )
    gone = Dataset(
        kind="pipe.passport", title="Паспорт PP-9999",
        error=ToolError(code="E-1108", message="Трубы нет в парке.", retriable=False),
    )

    assert "через 60 с" in stale.digest()
    assert "повторять запрос бессмысленно" in gone.digest()
    # Код обязателен: по нему в вики есть таблица с инструкцией.
    assert "E-1042" in stale.digest() and "E-1108" in gone.digest()


def test_an_error_is_not_an_empty_table() -> None:
    """Отказ доезжает до интерфейса как отказ, а не как «ничего не нашлось».

    Разница практическая: пустая таблица говорит «таких труб нет», отказ
    говорит «мы не смогли посмотреть». Второе — не ответ на вопрос, и
    выдавать его за первое нельзя.
    """
    shown = Dataset(
        kind="pipes.fleet", title="Парк",
        error=ToolError(code="network", message="Сервис не отвечает.", retriable=True),
    ).to_dict()

    assert shown["error"]["code"] == "network"
    assert shown["rows"] == []


def test_the_handle_is_stable_for_the_same_data() -> None:
    """Прыгающий номер таблицы сделал бы два одинаковых прогона несравнимыми."""
    assert fleet(5).handle == fleet(5).handle
    assert fleet(5).handle != fleet(6).handle


def test_a_page_says_it_is_not_the_beginning() -> None:
    """Иначе модель напишет «топ-5», показывая шестую-десятую строки.

    Заголовок таблицы будет говорить одно, содержимое другое, и заметит
    это только тот, кто пересчитает строки руками.

    Диапазон строк нужен обоим читателям, а предупреждение «это не начало»
    и подсказка «попросите следующие 5» — только агенту. Отвечающая модель,
    прочитав их во фрагменте, пересказывала их пользователю вместо ответа.
    """
    page = fleet(5, total=39, truncated=True)
    page.offset = 5
    page.hint = "Показаны строки 6–10 из 39: попросите «следующие 5»."
    digest = page.digest()
    note = page.agent_note()

    assert "с 6-й по 10-ю" in digest
    assert "не начало списка" in note
    assert "первые 5 записей остались выше" in note

    assert "не начало списка" not in digest
    assert page.hint and page.hint not in digest


def test_the_shell_offers_a_way_to_reach_the_hidden_rows() -> None:
    """Оболочка, объявляющая о скрытых данных, обязана давать способ их достать.

    Без этого она не описывает ограничение, а провоцирует обход: на
    просьбу «следующие 5» модели остаётся запросить первые десять и
    назвать словами строки с шестой по десятую — то самое переписывание
    таблицы, от которого мы уходили.
    """
    assert "offset" in Dataset.__dataclass_fields__
    assert fleet(5, total=39, truncated=True).to_dict()["offset"] == 0


def test_the_browser_hands_back_the_cursor_of_the_table_on_screen() -> None:
    """Иначе «следующие 5» — догадка, и агент угадывает неверно.

    Из живого прогона: человек получил строки 1–5, попросил следующие пять
    и получил строки 1–10, а потом 11–20. Метка страницы лежала в браузере
    и до агента не доехала. Курсор держит тот, кто листает; сервер диалогов
    не помнит, значит вернуть метку обязан клиент.
    """
    block = open_tables_block(
        [OpenTable("T98D4", "Топ-5 труб", offset=0, shown=5, total_found=39,
                   next_cursor="fleet|0.0000|5|5")]
    )

    assert "T98D4" in block
    assert "строки 1–5 из 39" in block
    assert "fleet|0.0000|5|5" in block
    # Уже показанное перезапрашивать незачем: человек это видит.
    assert "заново не запрашивай" in block


def test_a_table_without_a_next_page_says_so() -> None:
    """Молчание тут читается как «метка потерялась», и агент идёт искать её."""
    block = open_tables_block(
        [OpenTable("T0001", "Весь парк", offset=0, shown=39, total_found=39, next_cursor="")]
    )
    assert "Дальше строк нет." in block


def test_nothing_shown_means_no_block_at_all() -> None:
    """Пустая рамка «открытых таблиц нет» — это токены за сообщение ни о чём."""
    assert open_tables_block([]) == ""
