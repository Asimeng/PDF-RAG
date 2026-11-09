"""PDF -> clean, page-aware Markdown.

Uses PyMuPDF4LLM, which keeps headings, lists and tables as Markdown. We then
remove noise that hurts retrieval: running headers/footers ("Author manuscript",
"Page 3"), superscript citation markers and excess whitespace.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf
import pymupdf4llm

from ragapp.ingest import ocr
from ragapp.schemas import PageText, ParsedDocument

log = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Raised when a PDF cannot be read (corrupt, encrypted, empty)."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_SUP = re.compile(r"<sup>(.*?)</sup>", re.S)
# Superscripts that are citation markers ("14,15", "3-5") or footnote symbols.
# Everything else carries meaning in scientific text and is kept inline:
# HLA-DR<sup>lo</sup> -> HLA-DRlo, CD14<sup>+</sup> -> CD14+, Ca<sup>2+</sup> -> Ca2+.
_CITATION_SUP = re.compile(r"^(?:[\d\s,;\-–—]*\d[\d\s,;\-–—]*|[\s*†‡§]+)$")


def _keep_meaningful_superscript(m: re.Match) -> str:
    inner = m.group(1).strip()
    return "" if not inner or _CITATION_SUP.match(inner) else inner
_HTML = re.compile(r"</?(?:sub|span|br|b|i|u)[^>]*>", re.I)
_MULTI_BLANK = re.compile(r"\n{3,}")
_PICTURE = re.compile(r"\*\*==> picture .*? intentionally omitted <==\*\*", re.I)
# Text found inside figures (axis labels, panel letters, stray headers) - pure noise.
_PICTURE_TEXT = re.compile(r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _line_key(line: str) -> str:
    """Normalise a line so 'Page 3' / 'Page 14' and '1420 Cell 182' / 'Cell 182 1421'
    compare equal: drop digits, punctuation and Markdown markup."""
    no_numbers = re.sub(r"\S*\d\S*", " ", line.lower())  # '1419–1440.e1–e12', 'e3'
    return re.sub(r"[\W_]+", " ", no_numbers).strip()


def _edge_indices(lines: list[str], edge: int) -> set[int]:
    """Indices of the first/last `edge` non-empty lines (where headers/footers live)."""
    filled = [i for i, ln in enumerate(lines) if ln.strip()]
    return set(filled[:edge] + filled[-edge:])


def strip_repeated_lines(pages: list[str], min_fraction: float = 0.4, edge: int = 4) -> list[str]:
    """Drop short lines at the top/bottom of pages that repeat on many pages
    (running headers/footers), plus bare page numbers."""
    if len(pages) < 3:
        return pages
    split = [t.splitlines() for t in pages]
    counts: Counter[str] = Counter()
    for lines in split:
        counts.update({_line_key(lines[i]) for i in _edge_indices(lines, edge)
                       if len(lines[i].strip()) <= 100})
    threshold = max(3, int(len(pages) * min_fraction))
    noisy = {k for k, c in counts.items() if c >= threshold and k}
    cleaned = []
    for lines in split:
        edges = _edge_indices(lines, edge)
        keep = []
        for i, ln in enumerate(lines):
            if i in edges:
                key = _line_key(ln)
                if ln.strip() and not key and len(ln.strip()) <= 12:
                    continue  # a bare page number such as "14" or "- 14 -"
                if key in noisy and len(ln.strip()) <= 100:
                    continue
            keep.append(ln)
        cleaned.append("\n".join(keep))
    return cleaned


def clean_text(text: str) -> str:
    text = _SUP.sub(_keep_meaningful_superscript, text)
    text = _HTML.sub("", text)
    text = _PICTURE.sub("", text)
    text = _PICTURE_TEXT.sub("", text)
    text = _COMMENT.sub("", text)
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def parse_pdf(path: Path, ocr_mode: str = "auto", ocr_dpi: int = 300) -> ParsedDocument:
    """Read a PDF into cleaned page texts.

    ``ocr_mode`` "auto" OCRs scanned pages and tables saved as images (see
    ingest/ocr.py); "off" never OCRs."""
    path = Path(path)
    if not path.exists():
        raise PDFParseError(f"File not found: {path}")
    ocr_pages: dict[int, str] = {}
    unread_scans = 0  # scanned pages we could not OCR (engine not installed)
    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise PDFParseError("PDF is password-protected")
            num_pages = doc.page_count
            if num_pages == 0:
                raise PDFParseError("PDF has no pages")
            if ocr_mode != "off":
                if ocr.engine_available():
                    # Adds the recognised text to this in-memory copy only; the file is untouched.
                    for page in doc:
                        if kind := ocr.apply(page, ocr_mode, ocr_dpi, visible=True):
                            ocr_pages[page.number + 1] = kind
                else:
                    unread_scans = sum(1 for page in doc if ocr.is_scanned(page))
            meta_title = (doc.metadata or {}).get("title", "") or ""
            if not _looks_like_title(meta_title):
                meta_title = _title_from_fonts(doc)
            page_dicts = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False,
                                                 use_ocr=False)  # OCR was done above, selectively
    except PDFParseError:
        raise
    except Exception as exc:  # pymupdf raises many different error types
        raise PDFParseError(f"Could not read PDF: {exc}") from exc

    raw = [p.get("text", "") for p in page_dicts]
    cleaned = [clean_text(t) for t in strip_repeated_lines(raw)]
    pages = [PageText(page=i + 1, text=t) for i, t in enumerate(cleaned) if t]

    if sum(len(p.text) for p in pages) < 50:
        if unread_scans:
            raise PDFParseError("No extractable text found - this looks like a scanned PDF. "
                                "Install the OCR engine to read it: pip install rapidocr")
        if ocr_mode == "off":
            raise PDFParseError("No extractable text found - the PDF is probably scanned "
                                "images (OCR is switched off: RAG_OCR=off)")
        raise PDFParseError("No readable text found, even with OCR "
                            "(the pages may be blank, or the scan too blurry)")
    if unread_scans:
        log.warning("%s: %d scanned page(s) skipped - install the OCR engine (pip install "
                    "rapidocr) to read them", path.name, unread_scans)

    title = meta_title.strip() or _guess_title(pages) or path.stem
    sha = file_sha256(path)
    log.info("Parsed %s: %d pages%s, title=%r", path.name, num_pages,
             f" ({len(ocr_pages)} read with OCR)" if ocr_pages else "", title)
    return ParsedDocument(
        doc_id=sha[:16], file_name=path.name, title=title, num_pages=num_pages, pages=pages,
        ocr_pages=ocr_pages, unread_scanned_pages=unread_scans,
    )


def _looks_like_title(s: str) -> bool:
    """Reject metadata junk like 'XCRM102285_grabs 1..1' or 'Microsoft Word - doc1'."""
    s = s.strip()
    if len(s) < 10 or len(s.split()) < 3:
        return False
    if re.search(r"_|\.\.|\.docx?$|^microsoft word|^untitled", s, re.I):
        return False
    return True


def _title_from_fonts(doc: "pymupdf.Document", max_pages: int = 3) -> str:
    """The title is usually the largest text on the first page that has real content."""
    best_size, best_text = 0.0, ""
    for page in list(doc)[:max_pages]:
        sizes: dict[float, list[str]] = {}
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt:
                        sizes.setdefault(round(span["size"], 1), []).append(txt)
        if not sizes:
            continue
        # Body text = the size covering the most characters; a title must be clearly larger.
        body = max(sizes, key=lambda s: sum(len(t) for t in sizes[s]))
        for size, texts in sizes.items():
            joined = " ".join(texts)
            if (size > best_size and size >= body * 1.2 and _looks_like_title(joined)
                    and len(joined) <= 250):
                best_size, best_text = size, joined
    return re.sub(r"\s+", " ", best_text).strip()


def _guess_title(pages: list[PageText]) -> str:
    for line in pages[0].text.splitlines()[:15]:
        s = line.strip().lstrip("#").strip().strip("*").strip()
        if 10 <= len(s) <= 200:
            return s
    return ""
