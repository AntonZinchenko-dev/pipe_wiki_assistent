"""Нарезка документа на чанки по структуре.

Два требования раздела 3, каждое из которых даёт больше, чем любая правка
промпта:

1. Резать по структуре, а не по числу символов. Нарезка «каждые 500 символов»
   рвёт мысль посередине: чанк заканчивается на середине объяснения, а
   следующий начинается с обрывка, и модель получает на вход два огрызка
   вместо одного цельного фрагмента.
2. Вставить заголовок раздела в текст чанка. Фрагмент «Отключение кэша в
   продакшене запрещено» без заголовка не отвечает на вопрос «в каком
   сервисе», и поиск его по специфичному запросу не находит. С заголовком
   «TELEMETRY-GW / Как отключить кэш последних значений» — находит.

Заголовки в извлечённом из PDF тексте не помечены разметкой: строка «Эндпоинты»
ничем не отличается от абзаца. Поэтому заголовок определяем по признакам:
короткая строка, без завершающей точки, отделённая пустыми строками, часто с
заглавной буквы. Эвристика неточная — и это нормально, пока мы знаем, где она
врёт: `scripts/ingest.py --dump` печатает дерево заголовков, и его смотрят
глазами.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pypdf нужен только извлечению, чанкингу он не нужен
    from .extract import ExtractedDoc

MAX_CHUNK_CHARS = 1400   # мягкий предел: режем большой раздел на части
MIN_CHUNK_CHARS = 120    # слишком мелкие склеиваем со следующим
HEADING_MAX_CHARS = 90
SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")
LIST_OR_TABLE = re.compile(r"^\s*([•\-*–]|\d+\.|\||\d+\)|[a-zа-я]\))")
COLUMN_GAP = re.compile(r"\S {3,}\S")
# «4.2.3.1 Порядок выдачи», «7.1 Ветвление» — составной номер раздела.
SECTION_NUMBER = re.compile(r"^\d+(\.\d+)+\.?\s+\S")
# Признаки строки кода/конфигурации, а не заголовка.
CODE_LINE = re.compile(r"[=<>{}]|^/|^[A-Z][A-Z0-9_]{2,}$|::")
# Только число, возможно с единицей измерения: ячейка таблицы.
MEASURE_ONLY = re.compile(
    r"^\d+([.,]\d+)?\s*(%|с|мс|м|км|мин|ч|шт\.?|тс|атм|КБ|МБ|ГБ|дн\.?)?$", re.IGNORECASE
)
# Сколько подряд идущих коротких строк считаем таблицей. Две короткие строки
# подряд — это ещё абзац с коротким хвостом плюс заголовок; три и больше —
# уже таблица, разобранная извлечением по ячейке на строку.
TABLE_RUN_MIN = 3
TABLE_SHORT_RATIO = 0.5


@dataclass(slots=True)
class Chunk:
    doc_id: str
    ordinal: int
    heading_path: str          # «TELEMETRY-GW / Кэш последних значений»
    text: str                  # уже с заголовком в начале — так и уходит в эмбеддинг
    body: str                  # без заголовка, для показа человеку
    page_from: int
    page_to: int
    content_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.content_hash:
            # Хеш по СОДЕРЖИМОМУ, а не по позиции: тогда повторный запуск
            # индексации не создаёт дублей, даже если документ сдвинулся
            # (идемпотентность, раздел 3).
            # Заголовок входит в хеш обязательно. Без него два РАЗНЫХ раздела
            # с одинаковым телом («Не применимо.», «Нет», «—» — в регламентах
            # это норма) дают один и тот же хеш, а колонка content_hash
            # уникальна: второй раздел молча не попадает в индекс вообще, и
            # индексация при этом считает его новым и тратит на него вызов
            # модели. Позиция в хеш по-прежнему не входит — идемпотентность
            # держится на содержимом, а заголовок это содержимое, не позиция.
            digest = hashlib.sha256(
                f"{self.doc_id}\x1f{self.heading_path}\x1f{self.body}".encode()
            ).hexdigest()
            self.content_hash = digest[:32]


def body_line_width(lines: list[str]) -> int:
    """Типичная ширина строки тела документа.

    Абзац, перенесённый по ширине страницы, состоит из строк почти одинаковой
    длины — близкой к максимуму. Заголовок заметно короче. Эта разница
    надёжнее любых пустых строк: в извлечённом PDF пустых строк почти нет, а
    заголовок часто вплотную примыкает к следующей строке таблицы.
    """
    lengths = sorted(len(line.strip()) for line in lines if len(line.strip()) > 20)
    if not lengths:
        return 80
    return lengths[int(len(lengths) * 0.9) - 1] if len(lengths) > 4 else lengths[-1]


def looks_like_heading(line: str, *, body_width: int = 80) -> bool:
    """Похожа ли строка на заголовок раздела.

    В извлечённом из PDF тексте заголовки ничем не помечены: строка
    «Эндпоинты» отличается от абзаца только своей формой, а разметки нет.
    Поэтому признаки такие:

    - нет табуляции — она осталась там, где в PDF были колонки таблицы, и
      строка таблицы заголовком не является;
    - не заканчивается точкой, запятой, точкой с запятой или двоеточием;
    - начинается с заглавной буквы или с составного номера раздела;
    - не элемент списка;
    - внутри нет точки с пробелом — это признак обычного предложения;
    - заметно короче типичной строки тела (см. body_line_width).

    Путь к этому набору был такой: сначала я требовал непустую строку СЛЕДОМ —
    не распознался ни один заголовок, потому что в PDF после заголовка обычно
    пусто. Потом требовал пустые строки с двух сторон — снова ни одного,
    потому что pdftotext почти не оставляет пустых строк. Работает только
    форма плюс ширина. Ни одна из этих ошибок не была видна на синтетическом
    тексте — они нашлись прогоном по настоящему корпусу, и это ровно то, что
    раздел 3 чек-листа называет «просмотреть извлечённое глазами».
    """
    stripped = line.strip()
    if not stripped or len(stripped) > HEADING_MAX_CHARS:
        return False
    if "\t" in line:
        return False
    # Число с единицей измерения — это ячейка таблицы: «1 %», «120 %», «4 с».
    # Заголовком раздела такое не бывает никогда.
    if MEASURE_ONLY.match(stripped):
        return False
    if stripped.endswith((".", ",", ";", ":")):
        return False
    if stripped[0].islower():
        return False
    # Строка блока кода: переменная окружения, путь эндпоинта, шаблон с
    # фигурными скобками. Формально короткая и с заглавной, но заголовком не
    # является — иначе разделами становятся `TG_SNAPSHOT_CACHE=on` и
    # `/internal/snapshot/{rig_id}`.
    if CODE_LINE.search(stripped):
        return False
    # Нумерованный заголовок регламента («4.2.3.1 Порядок выдачи») формально
    # похож на элемент нумерованного списка, но номер у него составной.
    numbered = bool(SECTION_NUMBER.match(stripped))
    if not numbered and LIST_OR_TABLE.match(stripped):
        return False
    if not numbered and re.search(r"\.\s+\S", stripped):
        return False
    return len(stripped) < body_width * 0.72


def table_region_indices(lines: list[str], body_width: int) -> set[int]:
    """Индексы строк, попавших внутрь таблицы.

    Зачем это нужно. Извлечение из PDF библиотекой pypdf не сохраняет
    разделители колонок — каждая ячейка приезжает отдельной строкой. По форме
    ячейка «Положение талевого блока» ничем не отличается от заголовка
    раздела: короткая, с заглавной, без точки на конце. Из-за этого на живом
    индексе разделами стали `120 %`, `1`, `2` и `4` — то есть заголовок,
    который мы подмешиваем в текст чанка ради качества поиска, превратился в
    мусор и стал качество портить.

    Отличает таблицу от заголовка не форма отдельной строки, а плотность:
    таблица — это длинная череда коротких строк подряд. Первую строку такой
    череды оставляем кандидатом в заголовки (это как раз «Словарь тегов» или
    «Пороги оповещения», после которых таблица и начинается), все остальные
    исключаем.
    """
    threshold = body_width * TABLE_SHORT_RATIO
    positions = [index for index, line in enumerate(lines) if line.strip()]
    inside: set[int] = set()

    run: list[int] = []
    for index in positions:
        if len(lines[index].strip()) < threshold:
            run.append(index)
            continue
        if len(run) >= TABLE_RUN_MIN:
            inside.update(run[1:])
        run = []
    if len(run) >= TABLE_RUN_MIN:
        inside.update(run[1:])

    return inside


def heading_indices(lines: list[str], body_width: int) -> set[int]:
    """Индексы строк, которые мы считаем заголовками разделов.

    Форма плюс два контекстных запрета: строка внутри таблицы и строка сразу
    после двоеточия. Двоеточие означает «дальше содержимое того, что я только
    что назвал» — список, таблица, блок кода, — и первая строка этого
    содержимого заголовком не является.
    """
    in_table = table_region_indices(lines, body_width)
    previous_non_empty: str | None = None
    result: set[int] = set()

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if (
            index not in in_table
            and not (previous_non_empty or "").rstrip().endswith(":")
            and looks_like_heading(line, body_width=body_width)
        ):
            result.add(index)
        previous_non_empty = line

    return result


def split_sections(doc: "ExtractedDoc", *, title: str) -> list[tuple[str, str, int, int]]:
    """Документ -> список (путь заголовков, текст раздела, страница от, до).

    Если извлечение дало размеры шрифта — режем по ним: это признак из самого
    документа, а не догадка по форме строки. Если размеров нет (текст из
    письма, из чужого парсера, из txt) — работает резервный разбор по форме.
    """
    if doc.has_font_sizes:
        return _split_by_font(doc, title=title)
    return _split_by_shape(doc, title=title)


def _split_by_font(doc: "ExtractedDoc", *, title: str) -> list[tuple[str, str, int, int]]:
    from .extract import HEADING_SIZE_RATIO

    body_size = doc.body_size()
    heading_floor = body_size * HEADING_SIZE_RATIO

    sections: list[tuple[str, str, int, int]] = []
    current = title
    buffer: list[str] = []
    page_from = doc.pages[0].number if doc.pages else 1
    page_to = page_from

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current, body, page_from, page_to))
        buffer = []

    for page in doc.pages:
        for line in page.lines:
            is_heading = body_size > 0 and line.size >= heading_floor
            if is_heading:
                if buffer:
                    flush()
                # Строка самого крупного кегля — это название документа, и
                # дублировать его в пути раздела незачем.
                same_as_title = line.text in title or title in line.text
                current = title if same_as_title else f"{title} / {line.text}"
                page_from = page.number
            else:
                # Диапазон страниц раздела задают строки его ТЕЛА, а не
                # заголовки. Раньше page_to обновлялся один раз в конце
                # страницы — и flush() внутри страницы (а он случается на
                # каждом заголовке) закрывал раздел с page_to от ПРЕДЫДУЩЕЙ
                # страницы: ссылка на источник выходила перевёрнутой, «с. 3–2».
                # Если же обновлять до разбора строки, раздел прихватывает
                # страницу следующего заголовка.
                if not buffer:
                    page_from = page.number
                page_to = page.number
                buffer.append(line.text)

    flush()
    return sections


def _split_by_shape(doc: "ExtractedDoc", *, title: str) -> list[tuple[str, str, int, int]]:
    """Резервный разбор: размеров шрифта нет, ориентируемся на форму строки.

    Страницы склеиваются в один поток строк с пометкой номера страницы. Иначе
    абзац, переходящий на следующую страницу, начинается «с новой строки без
    предыдущей» и опознаётся как заголовок — ровно эта ошибка и была на первом
    прогоне.

    Этот путь заведомо слабее разбора по кеглю: он не различает ячейку таблицы
    и заголовок надёжно, а только статистически. Поэтому он и резервный.
    """
    flat: list[tuple[str, int]] = []
    for page in doc.pages:
        for line in page.text.splitlines():
            flat.append((line, page.number))

    all_lines = [line for line, _ in flat]
    width = body_line_width(all_lines)
    headings = heading_indices(all_lines, width)
    sections: list[tuple[str, str, int, int]] = []
    current_heading = title
    buffer: list[str] = []
    page_from = doc.pages[0].number if doc.pages else 1
    page_to = page_from

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_heading, body, page_from, page_to))
        buffer = []

    for index, (line, page_number) in enumerate(flat):
        if index in headings:
            if buffer:
                flush()
            current_heading = f"{title} / {line.strip()}"
            page_from = page_number
        else:
            buffer.append(line)
        page_to = page_number

    flush()
    return sections


def pack(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Режет длинный раздел по границам предложений, а не по символам."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(text):
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 > limit and current:
            parts.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_document(doc: ExtractedDoc, *, doc_id: str, title: str) -> list[Chunk]:
    """Разделы -> чанки. Длинные режутся по предложениям, безголовые склеиваются.

    Про склейку отдельно, потому что первая версия здесь была неверной. Она
    приклеивала любой короткий раздел к следующему — и короткий раздел терял
    свой заголовок, уезжая в чанк под чужим. То есть ломалась именно та
    связка «заголовок — содержимое», ради которой заголовок и подмешивается в
    текст чанка. Заметил это тест, а не корпус: на живых данных потеря
    выглядела бы просто как «почему-то не находится раздел про включение
    кэша».

    Теперь склеиваются только БЕЗЗАГОЛОВОЧНЫЕ куски — вступление документа до
    первого заголовка. Раздел со своим заголовком остаётся отдельным чанком
    любой длины: короткий раздел это нормально, а вот раздел под чужим
    названием — нет.
    """
    chunks: list[Chunk] = []
    ordinal = 0
    carry = ""

    for heading, body, page_from, page_to in split_sections(doc, title=title):
        body = f"{carry}\n{body}".strip() if carry else body
        carry = ""
        headingless = heading == title
        if headingless and len(body) < MIN_CHUNK_CHARS:
            carry = body
            continue

        for part in pack(body):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    ordinal=ordinal,
                    heading_path=heading,
                    # Заголовок физически входит в текст, который пойдёт в
                    # эмбеддинг и в ключевой индекс. Самый дешёвый способ
                    # поднять качество поиска из всех известных.
                    text=f"{heading}\n\n{part}",
                    body=part,
                    page_from=page_from,
                    page_to=page_to,
                )
            )
            ordinal += 1

    if carry:
        chunks.append(
            Chunk(doc_id=doc_id, ordinal=ordinal, heading_path=title,
                  text=f"{title}\n\n{carry}", body=carry, page_from=1, page_to=1)
        )
    return chunks
