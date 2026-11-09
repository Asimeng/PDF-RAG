"""Render a PDF page as a PNG, optionally highlighting a cited passage.

This lets the GUI *show* the evidence behind an answer. Users see the actual page
with the passage marked in yellow, so they can check the answer without opening
the PDF.

The stored chunk text was cleaned and converted to Markdown, so it no longer
matches the PDF text exactly. Instead of searching for the passage verbatim, we
*align* the two word sequences (like a diff tool does): runs of three or more
matching words are highlighted, and small gaps between them (a hyphenated word,
a superscript) are filled in. The highlights are then merged line by line.

Scanned pages (and tables saved as images) have no words to align with, so the
same OCR used during ingestion is run on the page first and its text is added as
an invisible layer. OCR results are cached per page, so this only costs time the
first time a page is shown.
"""

from __future__ import annotations

import functools
import re
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf

from ragapp.ingest import ocr

RENDER_OCR_DPI = 200  # enough for locating a passage; faster than ingestion's 300


@functools.lru_cache(maxsize=64)
def _cached_ocr_lines(pdf_path: str, mtime: float, page_index: int) -> tuple:
    with pymupdf.open(pdf_path) as doc:
        return tuple(ocr.page_lines(doc[page_index], RENDER_OCR_DPI))

_NON_WORD = re.compile(r"[^\w]+")
MIN_RUN = 3  # words in a row that must match before we trust a match
MAX_GAP = 4  # unmatched words allowed between two matched runs


def _norm(word: str) -> str:
    return _NON_WORD.sub("", word.lower())


def matched_word_indices(page_words: list[str], passage: str) -> list[int]:
    """Indices into ``page_words`` that belong to ``passage``."""
    # Align only real words; pure punctuation ("," "-") would break up runs.
    keep = [(i, n) for i, n in enumerate(_norm(w) for w in page_words) if n]
    pw = [n for _, n in keep]
    cw = [w for w in (_norm(t) for t in passage.split()) if w]
    if not cw or not pw:
        return []
    sm = SequenceMatcher(None, pw, cw, autojunk=False)
    hits = sorted(
        j
        for block in sm.get_matching_blocks()
        if block.size >= MIN_RUN
        for j in range(block.a, block.a + block.size)
    )
    filled: list[int] = []  # positions in pw
    for j in hits:
        if filled and 1 < j - filled[-1] <= MAX_GAP + 1:
            filled.extend(range(filled[-1] + 1, j))
        filled.append(j)
    if not filled:
        return []
    # Map back to page indices, including punctuation sitting inside the passage.
    first, last = keep[filled[0]][0], keep[filled[-1]][0]
    inside = {keep[j][0] for j in filled}
    return [i for i in range(first, last + 1)
            if i in inside or (not _norm(page_words[i]) and first < i < last)]


def render_page(pdf_path: Path, page_number: int, highlight: str | None = None,
                zoom: float = 1.6, ocr_mode: str = "auto") -> bytes:
    """Return PNG bytes of a 1-based page. Raises ValueError for a bad page number."""
    with pymupdf.open(pdf_path) as doc:
        if not 1 <= page_number <= doc.page_count:
            raise ValueError(f"Page {page_number} is outside 1-{doc.page_count}")
        page = doc[page_number - 1]
        if highlight and ocr_mode != "off" and ocr.engine_available():
            lines = _cached_ocr_lines(str(pdf_path), Path(pdf_path).stat().st_mtime, page_number - 1)
            ocr.insert_lines(page, list(lines), visible=False)  # invisible: image stays clean
        if highlight:
            words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
            lines: dict[tuple[int, int], pymupdf.Rect] = {}
            for i in matched_word_indices([w[4] for w in words], highlight):
                x0, y0, x1, y1, _, block, line, _ = words[i]
                key = (block, line)
                rect = pymupdf.Rect(x0, y0, x1, y1)
                lines[key] = lines[key] | rect if key in lines else rect
            for rect in lines.values():
                page.add_highlight_annot(rect)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True)
        return pix.tobytes("png")
