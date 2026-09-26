"""Извлечение текста из PDF и чистка того, что туда добавила вика.

Это первый шаг, на котором качество умирает молча. Код отработает без единой
ошибки, а в индекс уедут колонтитулы посреди предложения, разорванные слова с
переносами и таблицы, превращённые в поток ячеек без заголовков столбцов.

Главный урок этого файла достался дорого. Сначала заголовки разделов я
определял по форме строки: короткая, с заглавной, без точки на конце. Две
итерации подряд это давало неверный результат, а на живом индексе разделами
стали `120 %`, `1`, `2` и `4` — ячейки таблиц. Потом выяснилось, что PDF
несёт ответ прямо в себе: у заголовка другой размер шрифта. Извлечение с
размерами — `extract_pdf` — и есть правильный путь, а разбор по форме остался
только резервом для источников, где размеров нет.

Мораль общая: прежде чем строить эвристику поверх данных, стоит проверить, не
лежит ли нужный признак в самих данных.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

# Мягкий перенос и разрыв слова на границе строки: «инклиномет-\nрия».
HYPHEN_BREAK = re.compile(r"(\w)[­-]\n(\w)")
# Тот же перенос, но на конце ОТДЕЛЬНОЙ строки: между строками PDF символа
# \n нет, строки приходят списком.
HYPHEN_TAIL = re.compile(r"(?<=\w)[­-]$")
COLUMN_GAP = re.compile(r"[  ]{3,}")
MULTI_SPACE = re.compile(r"[  ]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")
PAGE_MARK = re.compile(r"^\s*(Стр\.|Страница|Page)\s*\d+(\s*(из|of)\s*\d+)?\s*$", re.IGNORECASE)
DIGITS = re.compile(r"\d+")

# Сколько строк с каждого края страницы считаются зоной колонтитула. Три —
# потому что типичная выгрузка из вики ставит туда организацию, идентификатор
# документа и пометку о конфиденциальности. Значение живёт в ОДНОМ месте:
# продублированное в двух функциях, оно однажды разошлось (в чтении PDF стояло
# два, в чистке три), и третья строка колонтитула тихо уезжала в индекс.
EDGE_ZONE_LINES = 3

# Во сколько раз заголовок крупнее основного текста. Порог низкий сознательно:
# в нашей вике разделы 13 pt против 10.5 pt текста, но в чужой выгрузке
# разница может быть меньше.
HEADING_SIZE_RATIO = 1.08


@dataclass(slots=True)
class Line:
    """Визуальная строка с размером шрифта.

    `size` — эффективный размер в точках, то есть размер шрифта, умноженный на
    вертикальный масштаб текстовой матрицы. Ноль означает «неизвестен»: так
    выглядят строки из источников без разметки шрифтов.
    """

    text: str
    size: float = 0.0


@dataclass(slots=True)
class ExtractedPage:
    number: int  # с единицы, как у человека
    text: str
    lines: list[Line] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedDoc:
    pages: list[ExtractedPage]
    dropped_lines: list[str]  # что выкинули как колонтитул — для проверки глазами

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def has_font_sizes(self) -> bool:
        return any(line.size > 0 for page in self.pages for line in page.lines)

    def body_size(self) -> float:
        """Размер шрифта основного текста — самый «многосимвольный» на документ.

        Именно по числу символов, а не по числу строк: ячеек таблицы в
        документе может быть больше, чем абзацев, но символов в абзацах всё
        равно кратно больше.
        """
        weights: Counter[float] = Counter()
        for page in self.pages:
            for line in page.lines:
                if line.size > 0:
                    weights[line.size] += len(line.text)
        return weights.most_common(1)[0][0] if weights else 0.0


def _page_lines(page) -> list[Line]:
    """Собирает строки страницы вместе с размером шрифта.

    pypdf вызывает visitor на каждой операции вывода текста и передаёт кусок
    текста вместе с текстовой матрицей. Ключевая тонкость: кусок уже может
    содержать переводы строк — первая версия этой функции группировала куски
    по вертикальной координате `tm[5]` и склеивала соседние строки в одну,
    из-за чего заголовки получались вида «конвенции\\nСтек и обоснование».

    Поэтому идём по потоку кусков в порядке вывода и режем его по переводам
    строк, запоминая самый крупный кегль внутри строки. Максимум, а не
    первый шрифт: в строке текста может встретиться вставка `кода` мельче
    основного, и от этого строка заголовком не становится — и наоборот.
    """
    lines: list[Line] = []
    parts: list[str] = []
    size = 0.0

    def flush() -> None:
        nonlocal parts, size
        raw = "".join(parts)
        # Разделитель колонок ищем ДО схлопывания пробелов: иначе три пробела
        # уже превратились в один, и след табличной разметки исчез. Резервное
        # распознавание заголовков (для PDF без размеров шрифта) опирается
        # именно на табуляцию — без этой строки оно не работало вообще, и
        # ячейки таблиц снова становились заголовками.
        raw = COLUMN_GAP.sub("\t", raw)
        text = MULTI_SPACE.sub(" ", raw).strip()
        if text:
            lines.append(Line(text=text, size=round(size, 1)))
        parts = []
        size = 0.0

    def visitor(text: str, _cm, tm, _font_dict, font_size) -> None:
        nonlocal size
        if not text:
            return
        try:
            scale = float(tm[3]) if tm and tm[3] else 1.0
            current = float(font_size) * scale if font_size else 0.0
        except (TypeError, ValueError, IndexError):
            current = 0.0

        for index, chunk in enumerate(text.split("\n")):
            if index > 0:
                flush()
            if chunk.strip():
                parts.append(chunk)
                size = max(size, current)

    page.extract_text(visitor_text=visitor)
    flush()
    return lines


def extract_pdf(path: Path, *, edge_zone_lines: int = EDGE_ZONE_LINES) -> ExtractedDoc:
    """Чтение PDF с размерами шрифта и чисткой колонтитулов."""
    reader = PdfReader(str(path))
    pages_lines = [_page_lines(page) for page in reader.pages]
    return _assemble(pages_lines, edge_zone_lines=edge_zone_lines)


def signature(line: str) -> str:
    """Подпись строки для сравнения между страницами.

    Две вещи, из-за которых наивное сравнение «строка в строку» не работает и
    колонтитулы благополучно уезжают в индекс:

    1. Номер страницы. «Стр. 1» и «Стр. 2» — разные строки, а колонтитул один
       и тот же. Поэтому все числа в подписи заменяются на решётку.
    2. Выравнивание. Извлечение сохраняет пробелы между левой и правой частью
       колонтитула, и их количество зависит от ширины содержимого страницы.
       Поэтому пробелы схлопываются.
    """
    return DIGITS.sub("#", " ".join(line.split()))


def _zone(count: int, requested: int) -> int:
    """Краевая зона не может занимать всю страницу.

    На настоящей странице сорок строк, и три строки сверху — это ровно
    колонтитул. Но на короткой странице три строки сверху и три снизу
    покрывают её целиком, и повторяющийся абзац тела уезжает в мусор вместе с
    колонтитулом. Ограничение — треть страницы с каждого края.
    """
    return min(requested, max(1, count // 3))


def _assemble(
    pages_lines: list[list[Line]], *, edge_zone_lines: int = EDGE_ZONE_LINES
) -> ExtractedDoc:
    non_empty = [lines for lines in pages_lines if lines]
    threshold = max(2, int(len(non_empty) * 0.6))

    # Колонтитул — строка, которая повторяется в КРАЕВОЙ зоне большинства
    # страниц. Оба уточнения важны: без краевой зоны под нож пойдёт
    # повторяющаяся формулировка из тела, без порога — случайное совпадение.
    counter: Counter[str] = Counter()
    for lines in non_empty:
        top = _zone(len(lines), edge_zone_lines)
        bottom = _zone(len(lines), edge_zone_lines)
        edge = lines[:top] + lines[-bottom:]
        counter.update({signature(line.text) for line in edge})

    boilerplate = {sig for sig, count in counter.items() if count >= threshold and sig}

    pages: list[ExtractedPage] = []
    dropped: list[str] = []
    number = 0

    for lines in pages_lines:
        if not lines:
            continue
        number += 1
        top = _zone(len(lines), edge_zone_lines)
        bottom = _zone(len(lines), edge_zone_lines)
        edge_indices = set(range(top)) | set(range(len(lines) - bottom, len(lines)))

        kept: list[Line] = []
        for index, line in enumerate(lines):
            # Номер страницы убираем где угодно, а не только в краевой зоне:
            # порядок строк в извлечённом тексте — это порядок отрисовки, а не
            # порядок на листе, и «нижний» колонтитул часто оказывается сверху.
            if PAGE_MARK.match(line.text):
                dropped.append(line.text)
                continue
            if index in edge_indices and signature(line.text) in boilerplate:
                dropped.append(line.text)
                continue
            kept.append(line)

        # Склейка переносов применяется к СТРОКАМ, а не только к page.text.
        #
        # В индекс идут именно строки: chunk_document читает page.lines. Пока
        # normalize применялся только к page.text, «инклиномет-\nрия»
        # доезжала до эмбеддинга и до FTS разорванной на два токена, а
        # `--dump` при этом печатал page.text, то есть ЧИСТЫЙ текст. Просмотр
        # глазами показывал не то, что лежит в индексе, — худший вид ошибки в
        # данных: она не видна ровно там, где её ищут.
        kept = _glue_hyphen_breaks(kept)

        pages.append(
            ExtractedPage(
                number=number,
                text="\n".join(line.text for line in kept),
                lines=kept,
            )
        )

    return ExtractedDoc(pages=pages, dropped_lines=sorted(set(dropped)))


def strip_boilerplate(
    raw_pages: list[str],
    *,
    header_zone_lines: int = EDGE_ZONE_LINES,
    footer_zone_lines: int = EDGE_ZONE_LINES,
) -> ExtractedDoc:
    """Та же чистка для источников БЕЗ размеров шрифта.

    Резервный путь: текст из письма, из выгрузки в txt, из чужого парсера.
    Размеры остаются нулевыми, и чанкинг сам переключится на разбор по форме
    строки.
    """
    pages_lines = [
        [Line(text=line.strip()) for line in page.splitlines() if line.strip()]
        for page in raw_pages
    ]
    return _assemble(pages_lines, edge_zone_lines=max(header_zone_lines, footer_zone_lines))


def _glue_hyphen_breaks(lines: list[Line]) -> list[Line]:
    """Склеивает слово, разорванное переносом между строками.

    Перенос в PDF — это разрыв МЕЖДУ строками, поэтому склеивать его надо на
    уровне списка строк: внутри одной строки его не видно. Кегль склеенной
    строки берём от первой — заголовки переносами не рвутся, так что выбор
    ни на что не влияет, но должен быть определённым.
    """
    glued: list[Line] = []
    for line in lines:
        if glued and HYPHEN_TAIL.search(glued[-1].text) and line.text[:1].isalpha():
            previous = glued[-1]
            glued[-1] = Line(
                text=HYPHEN_TAIL.sub("", previous.text) + line.text,
                size=previous.size,
            )
            continue
        glued.append(line)
    return glued


def normalize(text: str) -> str:
    """Приведение текста к виду, пригодному для чанкинга.

    Переносы склеиваем ДО схлопывания пробелов, иначе получится «инклиномет
    рия» с пробелом внутри слова — и такой чанк не найдётся ни вектором, ни
    ключевым поиском.

    Разделитель колонок (три и больше пробелов) становится табуляцией: это
    единственный след табличной разметки в тех источниках, где нет размеров
    шрифта, и по нему ориентируется резервное распознавание заголовков.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    text = COLUMN_GAP.sub("\t", text)
    text = MULTI_SPACE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
