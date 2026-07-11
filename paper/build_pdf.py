"""Render the canonical Markdown manuscript to a submission PDF (no pandoc/LaTeX).

The single source of truth is ``radiomics-phantom_medRxiv.md`` (pandoc-style
Markdown, matching the IORN house format). This script parses the controlled
subset of Markdown that manuscript uses and lays it out with reportlab, so the
PDF is regenerable from the manuscript in any environment. When pandoc is
available, ``pandoc radiomics-phantom_medRxiv.md -o radiomics-phantom_medRxiv.pdf``
is an equivalent path.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
MD = HERE / "radiomics-phantom_medRxiv.md"
OUT = HERE / "radiomics-phantom_medRxiv.pdf"


def png_size(path: Path) -> tuple[int, int]:
    w, h = struct.unpack(">II", path.read_bytes()[16:24])
    return w, h


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split ``--- yaml ---`` frontmatter from the body; parse the fields we use."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return {}, text
    fm_raw, body = m.group(1), m.group(2)
    meta: dict = {"author": ""}
    mode = None
    buf: list[str] = []
    for line in fm_raw.splitlines():
        key = re.match(r"^(\w+):\s*(.*)$", line)
        if key and not line.startswith(" "):
            if mode == "abstract":
                meta["abstract"] = " ".join(buf).strip()
                buf = []
            mode = None
            name, val = key.group(1), key.group(2).strip().strip('"')
            if name in ("title", "date", "keywords"):
                meta[name] = val
            elif name == "abstract" and val == "|":
                mode = "abstract"
            elif name == "author":
                mode = "author"
        elif mode == "abstract":
            buf.append(line.strip())
        elif mode == "author" and line.strip().startswith("-"):
            meta["author"] = line.strip().lstrip("- ").strip()
    if mode == "abstract":
        meta["abstract"] = " ".join(buf).strip()
    return meta, body


def inline(text: str) -> str:
    """Markdown inline -> reportlab markup for a controlled subset."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<u>\1</u>", text)  # links -> underlined text
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def main() -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=10,
        leading=13.5,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "h1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
    )
    h2 = ParagraphStyle(
        "h2",
        parent=styles["Heading2"],
        fontName="Helvetica-BoldOblique",
        fontSize=10,
        leading=12.5,
        spaceBefore=6,
        spaceAfter=2,
    )
    title = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, spaceAfter=3)
    caption = ParagraphStyle(
        "caption", parent=body, fontSize=8.5, leading=11, spaceBefore=3, spaceAfter=10
    )
    abstract = ParagraphStyle(
        "abstract", parent=body, fontSize=9, leading=12, leftIndent=12, rightIndent=12
    )
    ref = ParagraphStyle(
        "ref",
        parent=body,
        fontSize=8.5,
        leading=11,
        leftIndent=14,
        firstLineIndent=-14,
        spaceAfter=3,
    )

    meta, bodytext = parse_frontmatter(MD.read_text(encoding="utf-8"))
    story: list = []
    story.append(Paragraph(meta.get("title", ""), title))
    story.append(
        Paragraph(
            meta.get("author", ""),
            ParagraphStyle("au", parent=body, alignment=TA_CENTER, spaceAfter=2),
        )
    )
    story.append(
        Paragraph(meta.get("date", ""), ParagraphStyle("dt", parent=small, alignment=TA_CENTER))
    )
    story.append(Spacer(1, 6))

    lines = bodytext.splitlines()
    i = 0
    in_refs = False
    para: list[str] = []

    def flush() -> None:
        nonlocal para
        if para:
            story.append(Paragraph(inline(" ".join(para).strip()), body))
            para = []

    # Pre-section info blocks come before the first "## " heading.
    seen_section = False
    while i < len(lines):
        line = lines[i].rstrip()
        if line.strip() == "---":
            flush()
            i += 1
            continue
        if line.startswith("## "):
            flush()
            seen_section = True
            heading = line[3:].strip()
            in_refs = heading.lower().startswith("references")
            if in_refs:
                story.append(Paragraph("References", h1))
            else:
                # Insert Abstract + keywords right before section 1.
                if heading.startswith("1."):
                    story.append(Paragraph("Abstract", h1))
                    story.append(Paragraph(inline(meta.get("abstract", "")), abstract))
                    if meta.get("keywords"):
                        story.append(
                            Paragraph("<b>Keywords:</b> " + inline(meta["keywords"]), small)
                        )
                story.append(Paragraph(heading, h1))
            i += 1
            continue
        if line.startswith("### "):
            flush()
            story.append(Paragraph(line[4:].strip(), h2))
            i += 1
            continue
        img = re.match(r"^!\[(.*)\]\(([^)]+)\)$", line)
        if img:
            flush()
            cap, path = img.group(1), img.group(2)
            p = HERE / path
            w, h = png_size(p)
            width = 5.2 * inch
            story.append(Image(str(p), width=width, height=width * h / w))
            story.append(Paragraph(inline(cap), caption))
            i += 1
            continue
        if in_refs and re.match(r"^\d+\.\s", line):
            flush()
            story.append(Paragraph(inline(line), ref))
            i += 1
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        # A pre-section info block line (bold lead) becomes its own small paragraph.
        if not seen_section and line.startswith("**"):
            flush()
            story.append(Paragraph(inline(line), small))
            i += 1
            continue
        para.append(line.strip())
        i += 1
    flush()

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title=meta.get("title", ""),
        author=meta.get("author", ""),
    )
    doc.build(story)
    print("Wrote", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
