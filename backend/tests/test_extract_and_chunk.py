"""Тесты извлечения и чанкинга.

Все эти проверки написаны ПОСЛЕ прогона по настоящему корпусу — потому что
именно он нашёл ошибки, которых не было видно на синтетическом тексте:
колонтитулы не вычищались (номер страницы делал строки неодинаковыми), а
заголовки не распознавались (в извлечённом PDF почти нет пустых строк).
"""

from __future__ import annotations

from app.rag.chunk import body_line_width, chunk_document, looks_like_heading, pack
from app.rag.extract import (
    ExtractedDoc, ExtractedPage, Line, normalize, signature, strip_boilerplate,
)

HEADER = "ООО «ГеоСтек» · Отдел цифровых продуктов                    TG-GW · ред. 2.1"
FOOTER = "Внутренний документ. Не для передачи третьим лицам.            Стр. {n}"


def page(number: int, body: str, *, spacing: int = 20) -> str:
    """Страница как её отдаёт извлечение: колонтитул, тело, колонтитул.

    Параметр spacing моделирует то, из-за чего наивное сравнение строк не
    работает: выравнивание правой части колонтитула зависит от страницы.
    """
    header = HEADER.replace("                    ", " " * spacing)
    return f"{header}\n\n{body}\n\n{FOOTER.format(n=number)}"


# ------------------------------------------------------------- колонтитулы


def test_signature_ignores_page_number_and_spacing() -> None:
    assert signature("Внутренний документ.    Стр. 1") == signature(
        "Внутренний документ.            Стр. 27"
    )


def test_boilerplate_removed_from_every_page() -> None:
    pages = [page(1, "Первый абзац документа."), page(2, "Второй абзац документа.", spacing=34)]
    doc = strip_boilerplate(pages)

    assert len(doc.pages) == 2
    for extracted in doc.pages:
        assert "ГеоСтек" not in extracted.text
        assert "Не для передачи" not in extracted.text
    assert doc.dropped_lines


def test_repeated_body_line_is_not_treated_as_boilerplate() -> None:
    """Повторяющаяся формулировка в ТЕЛЕ документа — не колонтитул.

    Поэтому чистка смотрит только на краевые зоны страницы: иначе под нож
    пойдёт нормальный текст, встречающийся на каждой странице.
    """
    shared = "Отключение кэша в продакшене запрещено."
    pages = [
        page(1, f"Начало.\n{shared}\nСередина первой страницы, достаточно длинная строка."),
        page(2, f"Продолжение.\n{shared}\nСередина второй страницы, тоже длинная строка."),
    ]
    doc = strip_boilerplate(pages)
    assert sum(shared in extracted.text for extracted in doc.pages) == 2


def test_empty_trailing_page_is_dropped() -> None:
    doc = strip_boilerplate([page(1, "Текст."), ""])
    assert len(doc.pages) == 1


def test_three_line_boilerplate_removed_from_short_last_page() -> None:
    """Последняя страница короткая, а колонтитул на ней тот же — три строки.

    Пока краевая зона схлопывалась до шестой части страницы, на такой странице
    оставалась третья строка колонтитула. Ошибка нашлась прогоном по корпусу:
    в трёх документах из шестнадцати «Внутренний документ…» уезжал в индекс.
    """
    long_body = "\n".join(f"Строка тела номер {i} достаточной длины для абзаца." for i in range(30))
    short_body = "\n".join(f"Короткий хвост {i} на последней странице." for i in range(9))
    doc = strip_boilerplate([page(1, long_body), page(2, short_body, spacing=28)])

    for extracted in doc.pages:
        assert "ГеоСтек" not in extracted.text
        assert "Не для передачи" not in extracted.text
        assert "Стр." not in extracted.text


def test_normalize_glues_hyphen_break() -> None:
    assert "инклинометрия" in normalize("инклиномет-\nрия версионирована")


def test_normalize_keeps_column_gaps_as_tabs() -> None:
    """Разделитель колонок должен остаться различимым: без него строка таблицы
    неотличима от заголовка."""
    assert "\t" in normalize("bit_depth      м      Глубина долота")
    assert "\t" not in normalize("обычный текст с одиночными пробелами")


# ---------------------------------------------------------------- заголовки


def test_heading_recognised_by_form_and_width() -> None:
    assert looks_like_heading("Словарь тегов", body_width=75)
    assert looks_like_heading("Коды ошибок", body_width=75)


def test_numbered_section_is_heading_not_list_item() -> None:
    assert looks_like_heading("4.2.3.1 Порядок выдачи", body_width=75)
    assert looks_like_heading("7.1 Ветвление", body_width=75)


def test_body_line_is_not_heading() -> None:
    assert not looks_like_heading(
        "Отключать кэш штатно требуется в одном случае: при отладке источника",
        body_width=75,
    )
    assert not looks_like_heading("подрядчика, часть — файловые выгрузки.", body_width=75)
    assert not looks_like_heading("Труба списывается по категории SCRAP.", body_width=75)


def test_table_row_is_not_heading() -> None:
    assert not looks_like_heading("Внутренний тег\tЕдиница\tЗначение", body_width=75)


def test_code_line_is_not_heading() -> None:
    for line in ("TG_SNAPSHOT_CACHE=on", "/internal/snapshot/{rig_id}", "PW_TEMPERATURE=0"):
        assert not looks_like_heading(line, body_width=75), line


def test_list_item_is_not_heading() -> None:
    assert not looks_like_heading("• Первый пункт", body_width=75)
    assert not looks_like_heading("1. Зафиксировать", body_width=75)


def test_full_width_line_is_not_heading_even_if_short_enough() -> None:
    """Ширина решает: строка тела абзаца близка к максимуму, заголовок — нет."""
    line = "Кэш включён по умолчанию и переживает перезапуск"
    assert looks_like_heading(line, body_width=90)
    assert not looks_like_heading(line, body_width=52)


def test_body_line_width_takes_typical_not_maximum() -> None:
    lines = ["к" * 74, "к" * 75, "к" * 73, "к" * 76, "Заголовок", "к" * 74]
    assert 70 <= body_line_width(lines) <= 76


# ------------------------------------------------------------------ чанкинг


def test_chunking_splits_negation_pair_into_separate_sections() -> None:
    """Пара «включить/отключить» обязана попасть в РАЗНЫЕ чанки.

    Если они окажутся в одном, тест на ложных друзей теряет смысл: любой
    поиск «найдёт правильно», потому что оба ответа в одном фрагменте.
    """
    body = (
        "Как включить кэш\n"
        + "Кэш включён по умолчанию, включается переменной окружения. " * 4
        + "\nКак отключить кэш\n"
        + "Отключать требуется только при отладке источника на стенде. " * 4
    )
    doc = ExtractedDoc(pages=[ExtractedPage(1, body)], dropped_lines=[])
    chunks = chunk_document(doc, doc_id="TG-GW", title="TELEMETRY-GW")

    headings = [chunk.heading_path for chunk in chunks]
    assert any("включить" in heading for heading in headings)
    assert any("отключить" in heading for heading in headings)
    on = next(c for c in chunks if "включить" in c.heading_path)
    off = next(c for c in chunks if "отключить" in c.heading_path)
    assert on.chunk_id if hasattr(on, "chunk_id") else True
    assert "Отключать" not in on.body
    assert "по умолчанию" not in off.body


def test_heading_is_part_of_chunk_text() -> None:
    """Заголовок физически входит в текст, который уйдёт в эмбеддинг: самый
    дешёвый способ поднять качество поиска."""
    doc = ExtractedDoc(
        pages=[ExtractedPage(1, "Коды ошибок\n" + "Код E-1042 означает устаревший расчёт. " * 5)],
        dropped_lines=[],
    )
    chunk = chunk_document(doc, doc_id="FA-API", title="FATIGUE-API")[0]
    assert chunk.text.startswith(chunk.heading_path)
    assert chunk.heading_path not in chunk.body


def test_content_hash_ignores_position() -> None:
    """Хеш по содержимому, а не по позиции: иначе сдвиг документа на одну
    строку создаст дубли всего документа при переиндексации.

    Вставка сверху взята заведомо большой: короткий раздел по правилу
    MIN_CHUNK_CHARS приклеивается к следующему, и тогда изменится уже само
    содержимое — а сравнивать надо одинаковое содержимое на разных позициях.
    """
    doc_a = ExtractedDoc(pages=[ExtractedPage(1, "Раздел\n" + "Текст раздела. " * 20)], dropped_lines=[])
    doc_b = ExtractedDoc(
        pages=[ExtractedPage(1, "Вставка сверху. " * 30 + "\n\nРаздел\n" + "Текст раздела. " * 20)],
        dropped_lines=[],
    )
    hashes_a = {chunk.content_hash for chunk in chunk_document(doc_a, doc_id="D", title="T")}
    hashes_b = {chunk.content_hash for chunk in chunk_document(doc_b, doc_id="D", title="T")}
    assert hashes_a & hashes_b


def test_pack_splits_on_sentence_boundaries() -> None:
    text = " ".join(f"Предложение номер {i} про усталость трубы." for i in range(60))
    parts = pack(text, limit=400)
    assert len(parts) > 1
    assert all(len(part) <= 460 for part in parts)
    assert all(part.endswith(".") for part in parts)


# ------------------------------------------------- разбор по размеру шрифта


def test_body_size_is_the_most_frequent_by_characters() -> None:
    """Размер тела определяется по числу СИМВОЛОВ, а не строк.

    Ячеек таблицы в документе бывает больше, чем абзацев, но символов в
    абзацах кратно больше — иначе за основной текст был бы принят кегль
    таблицы, и заголовками стали бы все абзацы подряд.
    """
    doc = ExtractedDoc(
        pages=[
            ExtractedPage(
                1,
                "",
                [
                    Line("Заголовок раздела", 13.0),
                    Line("Длинный абзац основного текста документа, который занимает всю строку", 10.5),
                    Line("Ещё один такой же длинный абзац основного текста этого документа", 10.5),
                    Line("S1", 8.6),
                    Line("S2", 8.6),
                    Line("S3", 8.6),
                    Line("S4", 8.6),
                    Line("1 %", 8.6),
                ],
            )
        ],
        dropped_lines=[],
    )
    assert doc.has_font_sizes
    assert doc.body_size() == 10.5


def test_font_split_uses_heading_size_and_ignores_table_cells() -> None:
    """Главная проверка шага 2: ячейка таблицы больше не заголовок.

    На живом индексе разделами стали `120 %`, `1`, `2` и `4` — ячейки из
    таблицы порогов. Признак «крупнее основного текста» отсекает их без
    единой эвристики про форму строки.
    """
    body = "Оповещение поднимается автоматически по следующим порогам без участия человека"
    doc = ExtractedDoc(
        pages=[
            ExtractedPage(
                1,
                "",
                [
                    Line("Регламент инцидентов РЛ-9.2", 17.0),
                    Line("Пороги оповещения", 13.0),
                    Line(body, 10.5),
                    Line("Метрика", 8.6),
                    Line("Порог внимания", 8.6),
                    Line("120 %", 8.6),
                    Line("1", 8.6),
                    Line("Дежурство", 13.0),
                    Line("Дежурит один человек в неделю, смена с вторника, и это не льгота", 10.5),
                ],
            )
        ],
        dropped_lines=[],
    )
    chunks = chunk_document(doc, doc_id="RL-92", title="Регламент инцидентов РЛ-9.2")
    headings = [chunk.heading_path for chunk in chunks]

    assert any(heading.endswith("Пороги оповещения") for heading in headings)
    assert any(heading.endswith("Дежурство") for heading in headings)
    for junk in ("120 %", "Метрика", "Порог внимания"):
        assert not any(heading.endswith(junk) for heading in headings), junk
    # Содержимое таблицы при этом не потеряно — оно в разделе про пороги.
    thresholds = next(c for c in chunks if c.heading_path.endswith("Пороги оповещения"))
    assert "120 %" in thresholds.body


def test_title_line_does_not_duplicate_itself_in_heading_path() -> None:
    doc = ExtractedDoc(
        pages=[
            ExtractedPage(
                1,
                "",
                [
                    Line("Регламент релизов РЛ-7.1", 17.0),
                    Line("Основная ветка main, работа ведётся в ветках от неё с префиксом по типу", 10.5),
                ],
            )
        ],
        dropped_lines=[],
    )
    chunk = chunk_document(doc, doc_id="RL-71", title="Регламент релизов РЛ-7.1")[0]
    assert chunk.heading_path == "Регламент релизов РЛ-7.1"


def test_falls_back_to_shape_when_font_sizes_are_absent() -> None:
    """Источник без размеров шрифта (txt, письмо, чужой парсер) обязан
    продолжать работать — на резервном разборе по форме строки."""
    doc = ExtractedDoc(
        pages=[ExtractedPage(1, "Коды ошибок\n" + "Код E-1042 означает устаревший расчёт. " * 6)],
        dropped_lines=[],
    )
    assert not doc.has_font_sizes
    chunks = chunk_document(doc, doc_id="FA-API", title="FATIGUE-API")
    assert chunks and chunks[0].heading_path.endswith("Коды ошибок")
