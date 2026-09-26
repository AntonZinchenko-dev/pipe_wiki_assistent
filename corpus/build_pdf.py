#!/usr/bin/env python3
"""Сборка корпуса: markdown-источники -> PDF, как их выгружает настоящая вики.

Зачем PDF, если рядом лежит чистый markdown. Ровно затем, что в жизни корпус
приходит именно так: выгрузкой из Confluence, из вики, из СЭД. И приходит он с
колонтитулами, с разорванными между страницами таблицами и с переносами
посреди предложения. Индексировать чистый источник — значит отлаживать поиск
на данных, которых в проде не будет.

Markdown-источники в corpus/source остаются не как «настоящий» корпус, а как
эталон: с ними сравнивают то, что извлеклось из PDF, когда глазами проверяют
качество извлечения (раздел 3 чек-листа).

Зависимость одна: pip install reportlab
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, ListFlowable, ListItem, PageTemplate,
    Paragraph, Preformatted, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
PDF_DIR = ROOT / "pdf"
MANIFEST = ROOT / "manifest.json"

ORG = "ООО «ГеоСтек» · Отдел цифровых продуктов"
CONFIDENTIAL = "Внутренний документ. Не для передачи третьим лицам."

PAGE_W, PAGE_H = A4
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 18 * mm, 22 * mm, 20 * mm

# Шрифты: DejaVu есть в linux-образах и в пакете matplotlib, на Windows
# обычно нет — тогда берём системный с кириллицей.
FONT_CANDIDATES = {
    "body": ["DejaVuSans.ttf", "arial.ttf", "segoeui.ttf", "calibri.ttf"],
    "bold": ["DejaVuSans-Bold.ttf", "arialbd.ttf", "segoeuib.ttf", "calibrib.ttf"],
    "italic": ["DejaVuSans-Oblique.ttf", "ariali.ttf", "segoeuii.ttf", "calibrii.ttf"],
    "mono": ["DejaVuSansMono.ttf", "consola.ttf", "cour.ttf"],
}
FONT_DIRS = [
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("C:/Windows/Fonts"),
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
]


def register_fonts() -> dict[str, str]:
    try:  # matplotlib таскает DejaVu с собой — удобный резерв на Windows
        import matplotlib
        FONT_DIRS.append(Path(matplotlib.get_data_path()) / "fonts/ttf")
    except Exception:
        pass

    names = {}
    for role, candidates in FONT_CANDIDATES.items():
        for candidate in candidates:
            found = next((d / candidate for d in FONT_DIRS if (d / candidate).exists()), None)
            if found:
                name = f"wiki-{role}"
                pdfmetrics.registerFont(TTFont(name, str(found)))
                names[role] = name
                break
        else:
            sys.exit(
                f"не найден шрифт для «{role}»: ни один из {candidates} нет в "
                f"{[str(d) for d in FONT_DIRS]}. Поставьте DejaVu или matplotlib."
            )
    return names


def styles(fonts: dict[str, str]) -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "body", fontName=fonts["body"], fontSize=10.5, leading=15.5,
        alignment=TA_JUSTIFY, spaceAfter=5,
    )
    return {
        "body": base,
        "h1": ParagraphStyle("h1", base, fontName=fonts["bold"], fontSize=17, leading=21,
                             alignment=0, spaceBefore=0, spaceAfter=10),
        "h2": ParagraphStyle("h2", base, fontName=fonts["bold"], fontSize=13, leading=17,
                             alignment=0, spaceBefore=12, spaceAfter=5),
        "h3": ParagraphStyle("h3", base, fontName=fonts["bold"], fontSize=11.5, leading=15,
                             alignment=0, spaceBefore=9, spaceAfter=4),
        "cell": ParagraphStyle("cell", base, fontSize=8.6, leading=11.5, alignment=0, spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", base, fontName=fonts["bold"], fontSize=8.6,
                                   leading=11.5, alignment=0, spaceAfter=0),
        "quote": ParagraphStyle("quote", base, leftIndent=10, textColor=colors.HexColor("#444444"),
                                borderPadding=0),
        "code": ParagraphStyle("code", base, fontName=fonts["mono"], fontSize=8.5, leading=11,
                               alignment=0, backColor=colors.HexColor("#f4f4f4"),
                               borderPadding=5, spaceBefore=4, spaceAfter=6),
    }


def inline(text: str, fonts: dict[str, str]) -> str:
    """markdown-разметка внутри строки -> разметка reportlab.

    Порядок важен: сначала экранируем всё, потом вставляем свои теги, иначе
    экранирование съест их же.
    """
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`",
                  rf'<font face="{fonts["mono"]}" size="9">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", rf'<font face="{fonts["bold"]}">\1</font>', text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])",
                  rf'<font face="{fonts["italic"]}">\1</font>', text)
    return text


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Полноценный YAML тут не нужен: в шапке только плоские пары."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[3:end].splitlines():
        if line.strip() and ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, text[end + 4 :].lstrip("\n")


def table_flowable(rows: list[list[str]], st: dict, fonts: dict[str, str]) -> Table:
    width = PAGE_W - 2 * MARGIN_X
    cols = max(len(r) for r in rows)
    rows = [r + [""] * (cols - len(r)) for r in rows]

    # Ширину колонки ведём от самой длинной ячейки, но не даём одной колонке
    # съесть страницу: иначе таблица кодов ошибок вырождается в две полосы.
    weights = [max(len(r[i]) for r in rows) ** 0.75 for i in range(cols)]
    total = sum(weights) or 1
    widths = [max(width * 0.09, width * w / total) for w in weights]
    scale = width / sum(widths)
    widths = [w * scale for w in widths]

    data = [[Paragraph(inline(c, fonts), st["cellhead" if i == 0 else "cell"]) for c in row]
            for i, row in enumerate(rows)]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ececec")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return table


def markdown_to_flowables(body: str, st: dict, fonts: dict[str, str]) -> list:
    flow: list = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):  # блок кода
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            flow.append(Preformatted("\n".join(buf), st["code"]))
            continue

        if stripped.startswith("|"):  # таблица
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):  # строка-разделитель
                    rows.append(cells)
                i += 1
            if rows:
                flow.append(Spacer(1, 4))
                flow.append(table_flowable(rows, st, fonts))
                flow.append(Spacer(1, 6))
            continue

        if match := re.match(r"^(#{1,3})\s+(.*)$", stripped):  # заголовок
            level = len(match.group(1))
            style = st[f"h{level}"]
            heading = Paragraph(inline(match.group(2), fonts), style)
            # Заголовок не должен оставаться один в конце страницы — тогда
            # извлечённый текст теряет связь раздела с его содержимым.
            flow.append(heading if level == 1 else KeepTogether([heading, Spacer(1, 1)]))
            i += 1
            continue

        if stripped.startswith(">"):  # цитата
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            flow.append(Paragraph(inline(" ".join(buf), fonts), st["quote"]))
            continue

        if re.match(r"^([-*]|\d+\.)\s+", stripped):  # список
            items, ordered = [], bool(re.match(r"^\d+\.", stripped))
            while i < len(lines) and re.match(r"^([-*]|\d+\.)\s+", lines[i].strip()):
                buf = [re.sub(r"^([-*]|\d+\.)\s+", "", lines[i].strip())]
                i += 1
                while i < len(lines) and lines[i].startswith(("  ", "\t")) and lines[i].strip():
                    buf.append(lines[i].strip())
                    i += 1
                items.append(ListItem(Paragraph(inline(" ".join(buf), fonts), st["body"]),
                                      leftIndent=16))
            flow.append(ListFlowable(items, bulletType="1" if ordered else "bullet",
                                     bulletFontName=fonts["body"], start="1" if ordered else None,
                                     leftIndent=16, bulletFontSize=9))
            flow.append(Spacer(1, 4))
            continue

        buf = []  # обычный абзац
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#{1,3}\s|\||>|```|[-*]\s|\d+\.\s)", lines[i].strip()
        ):
            buf.append(lines[i].strip())
            i += 1
        flow.append(Paragraph(inline(" ".join(buf), fonts), st["body"]))

    return flow


def build_one(md_path: Path, fonts: dict[str, str], st: dict) -> dict:
    meta, body = parse_frontmatter(md_path.read_text(encoding="utf-8"))
    doc_id = meta.get("doc_id", md_path.stem)
    version = meta.get("version", "—")
    pdf_path = PDF_DIR / f"{md_path.stem}.pdf"

    def decorate(canvas, doc):
        """Колонтитулы. В извлечённом тексте они окажутся между абзацами —
        это и есть та грязь, которую в разделе 3 чистят до чанкинга."""
        canvas.saveState()
        canvas.setFont(fonts["body"], 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        top = PAGE_H - MARGIN_TOP + 8
        canvas.drawString(MARGIN_X, top, ORG)
        canvas.drawRightString(PAGE_W - MARGIN_X, top, f"{doc_id} · ред. {version}")
        canvas.setStrokeColor(colors.HexColor("#cccccc"))
        canvas.line(MARGIN_X, top - 3, PAGE_W - MARGIN_X, top - 3)
        bottom = MARGIN_BOTTOM - 12
        canvas.drawString(MARGIN_X, bottom, CONFIDENTIAL)
        canvas.drawRightString(PAGE_W - MARGIN_X, bottom, f"Стр. {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=MARGIN_X, rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
        title=meta.get("title", md_path.stem), author=meta.get("owner", ORG),
        subject=f"{meta.get('project', '')} · {doc_id}",
    )
    frame = Frame(MARGIN_X, MARGIN_BOTTOM, PAGE_W - 2 * MARGIN_X,
                  PAGE_H - MARGIN_TOP - MARGIN_BOTTOM, id="body")
    doc.addPageTemplates([PageTemplate(id="wiki", frames=[frame], onPage=decorate)])
    doc.build(markdown_to_flowables(body, st, fonts))

    return {
        "doc_id": doc_id,
        "title": meta.get("title", md_path.stem),
        "project": meta.get("project", "Общее"),
        "owner": meta.get("owner", ""),
        "updated": meta.get("updated", ""),
        "version": version,
        "status": meta.get("status", "действующий"),
        "source": f"source/{md_path.name}",
        "pdf": f"pdf/{pdf_path.name}",
    }


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    fonts = register_fonts()
    st = styles(fonts)

    docs = []
    for md_path in sorted(SOURCE_DIR.glob("*.md")):
        entry = build_one(md_path, fonts, st)
        docs.append(entry)
        print(f"{entry['doc_id']:<12} {entry['pdf']}")

    MANIFEST.write_text(
        json.dumps({"org": ORG, "documents": docs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n{len(docs)} документов, манифест: {MANIFEST.name}")


if __name__ == "__main__":
    main()
