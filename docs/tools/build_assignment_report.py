"""Render the Markdown assignment report with bundled Python + ReportLab.

The source is the editable deliverable. This helper only generates its PDF.
Uses Arial/Consolas from Windows; pass --font-dir when needed on another host.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Preformatted,
    Spacer, Table, TableStyle,
)


NAVY = colors.HexColor("#193448")
TEAL = colors.HexColor("#176B70")
INK = colors.HexColor("#202D35")
GRAY = colors.HexColor("#657582")
LIGHT = colors.HexColor("#F2F5F7")
RULE = colors.HexColor("#D6DFE5")


def register_fonts(root: Path):
    fonts = {
        "Arial": "arial.ttf", "Arial-Bold": "arialbd.ttf",
        "Arial-Italic": "ariali.ttf", "Arial-BoldItalic": "arialbi.ttf",
        "Consolas": "consola.ttf",
    }
    for name, filename in fonts.items():
        pdfmetrics.registerFont(TTFont(name, str(root / filename)))
    pdfmetrics.registerFontFamily(
        "Arial", normal="Arial", bold="Arial-Bold",
        italic="Arial-Italic", boldItalic="Arial-BoldItalic",
    )


def inline(text: str) -> str:
    text = html.escape(text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Consolas">\1</font>', text)
    return text


def styles():
    common = dict(fontName="Arial", textColor=INK, alignment=TA_LEFT)
    return {
        "body": ParagraphStyle("Body", fontSize=10.6, leading=15.0,
                               spaceAfter=7, **common),
        "bullet": ParagraphStyle("Bullet", fontSize=10.6, leading=14.5,
                                 leftIndent=12, firstLineIndent=-9,
                                 spaceAfter=5, **common),
        "title": ParagraphStyle("Title", fontName="Arial-Bold", fontSize=25,
                                leading=31, textColor=colors.black, spaceAfter=18),
        "h1": ParagraphStyle("H1", fontName="Arial-Bold", fontSize=19,
                             leading=24, textColor=NAVY, spaceAfter=14,
                             keepWithNext=True),
        "h2": ParagraphStyle("H2", fontName="Arial-Bold", fontSize=12.2,
                             leading=17, textColor=TEAL, spaceBefore=9,
                             spaceAfter=7, keepWithNext=True),
        "cell": ParagraphStyle("Cell", fontSize=9.4, leading=12.9, **common),
        "head": ParagraphStyle("Head", fontName="Arial-Bold", fontSize=9.4,
                               leading=12.8, textColor=colors.white),
        "code": ParagraphStyle("Code", fontName="Consolas", fontSize=8.5,
                               leading=11.6, textColor=INK, leftIndent=8,
                               spaceBefore=5, spaceAfter=8),
    }


def table_block(rows: list[list[str]], width: float, ss: dict):
    n = len(rows[0])
    ratios = {2: [0.32, 0.68], 3: [0.24, 0.34, 0.42],
              4: [0.16, 0.27, 0.28, 0.29]}.get(n, [1 / n] * n)
    if n == 3 and "Giờ" in rows[0][1]:
        ratios = [0.49, 0.14, 0.37]
    data = [[Paragraph(inline(c), ss["head"] if i == 0 else ss["cell"])
             for c in row] for i, row in enumerate(rows)]
    table = Table(data, colWidths=[width * r for r in ratios],
                  repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), 0.35, RULE),
    ]))
    return [table, Spacer(1, 9)]


def parse_page(text: str, page_index: int, width: float, ss: dict):
    lines = text.strip().splitlines()
    flowables = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            flowables.append(Preformatted("\n".join(code), ss["code"]))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+", c) for c in cells):
                    rows.append(cells)
                i += 1
            flowables.extend(table_block(rows, width, ss))
        elif line.startswith("# "):
            flowables.append(Paragraph(inline(line[2:]), ss["title"] if page_index == 0 else ss["h1"]))
            i += 1
        elif line.startswith("## "):
            flowables.append(Paragraph(inline(line[3:]), ss["h2"]))
            i += 1
        elif line.startswith("- ") or re.match(r"^\d+\. ", line):
            if line.startswith("- "):
                value = "• " + line[2:]
            else:
                value = line
            flowables.append(Paragraph(inline(value), ss["bullet"]))
            i += 1
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not (
                lines[i].strip().startswith(("#", "|", "- ", "```"))
                or re.match(r"^\d+\. ", lines[i].strip())
            ):
                paragraph.append(lines[i].strip())
                i += 1
            flowables.append(Paragraph(inline(" ".join(paragraph)), ss["body"]))
    return flowables


def header_footer(canvas, doc):
    w, h = A4
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(44, h - 27, "FASHION VISUAL SEARCH")
    canvas.drawRightString(w - 44, h - 27, "Giao việc theo module | 28/10/2026")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.45)
    canvas.line(44, h - 35, w - 44, h - 35)
    canvas.line(44, 34, w - 44, 34)
    canvas.drawString(44, 21, "Tài liệu điều phối nội bộ | Phiên bản 1.0")
    canvas.drawRightString(w - 44, 21, str(doc.page))
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()
    register_fonts(args.font_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = args.source.read_text(encoding="utf-8-sig")
    # Normalize nonbreaking hyphens/dashes for copyable technical text.
    text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    sections = text.split("<!-- pagebreak -->")
    w, h = A4
    doc = BaseDocTemplate(str(args.output), pagesize=A4,
                          title="Báo cáo giao việc cải tiến Fashion Visual Search",
                          author="Nhóm Fashion Visual Search", leftMargin=44,
                          rightMargin=44, topMargin=49, bottomMargin=46)
    frame = Frame(44, 46, w - 88, h - 95, leftPadding=0,
                  rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="Report", frames=frame, onPage=header_footer))
    ss = styles()
    story = []
    for i, section in enumerate(sections):
        if i:
            story.append(PageBreak())
        story.extend(parse_page(section, i, w - 88, ss))
    doc.build(story)
    print(f"Rendered {len(sections)} source sections: {args.output}")


if __name__ == "__main__":
    main()
