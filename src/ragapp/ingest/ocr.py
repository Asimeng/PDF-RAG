"""OCR (optical character recognition) for text that has no text layer.

Most PDFs store their text as characters, which we read directly. Two common
cases don't:

  * **scanned pages**: the whole page is a photo of paper;
  * **tables pasted as images** inside an otherwise normal PDF.

For those we "read" the pixels with RapidOCR, a small ONNX OCR engine that runs
on CPU. The recognised text is written onto the page as an *invisible* text
layer (the technique "searchable PDF" scanners use), so the rest of the parser
handles it exactly like normal text.

Deliberately NOT OCR'd: figures, charts and photos. OCR of a figure produces
axis labels and tick numbers (noise for search) and was measured to make
parsing 5-12x slower on scientific papers. To tell a table image from a figure
without a heavy model we use its structure: tables have several horizontal
rules spanning (nearly) their full width and are drawn in black and white;
figures are usually coloured and rarely have three full-width rules.

Setting RAG_OCR: "auto" (default, as above) or "off".
"""

from __future__ import annotations

import importlib.util
import logging
import threading

import pymupdf

log = logging.getLogger(__name__)

MIN_PAGE_CHARS = 50  # fewer real characters than this = "no text layer"
SCANNED_IMAGE_FRACTION = 0.5  # ...and images cover at least half of the page
MIN_IMAGE_FRACTION = 0.04  # ignore images smaller than this share of the page (logos, icons)
TABLE_MIN_RULES = 3  # horizontal lines spanning >= 60% of the image width
TABLE_MAX_COLOURFUL = 0.05  # share of strongly coloured pixels allowed in a table image
DETECT_DPI = 150  # resolution for the cheap table/figure check
MIN_CONFIDENCE = 0.5  # drop OCR lines the engine is unsure about

_FONTNAME = "ragocr"
_FONT = pymupdf.Font("cjk")  # wide character coverage (Latin, Greek, CJK, symbols)
_engine = None
_engine_lock = threading.Lock()


def engine_available() -> bool:
    return importlib.util.find_spec("rapidocr") is not None


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr import RapidOCR

            log.info("Loading OCR engine (RapidOCR)...")
            _engine = RapidOCR(params={"Global.log_level": "warning"})  # keep its chatter quiet
    return _engine


def _pixmap_array(page: pymupdf.Page, rect: pymupdf.Rect, dpi: int):
    import numpy as np

    pix = page.get_pixmap(clip=rect, dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3), pix


def _large_images(page: pymupdf.Page) -> list[pymupdf.Rect]:
    page_area = abs(page.rect)
    rects = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if not r.is_empty and abs(r) >= MIN_IMAGE_FRACTION * page_area:
            rects.append(r)
    return rects


def is_scanned(page: pymupdf.Page) -> bool:
    """Little or no real text, and images cover most of the page."""
    text = page.get_text().strip()
    if len(text) - text.count("�") >= MIN_PAGE_CHARS:
        return False
    covered = pymupdf.Rect()
    area = 0.0
    for r in _large_images(page):
        area += abs(r)
        covered |= r
    # Use the union's area, capped by the sum (overlapping images aren't double counted).
    return min(area, abs(covered)) >= SCANNED_IMAGE_FRACTION * abs(page.rect)


def looks_like_table_image(page: pymupdf.Page, rect: pymupdf.Rect) -> bool:
    """Cheap pixel check: several full-width horizontal rules on a grey image."""
    import cv2
    import numpy as np

    img, pix = _pixmap_array(page, rect, DETECT_DPI)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if float((hsv[..., 1] > 60).mean()) > TABLE_MAX_COLOURFUL:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 15, -2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(pix.w * 0.6)), 1))
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    rows = np.where(rules.max(axis=1) > 0)[0]
    n_rules = 0 if len(rows) == 0 else 1 + int((np.diff(rows) > 3).sum())
    return n_rules >= TABLE_MIN_RULES


def plan(page: pymupdf.Page) -> tuple[str, list[pymupdf.Rect]]:
    """What to OCR on this page: ("page", [whole page]), ("table", regions) or ("", [])."""
    if is_scanned(page):
        return "page", [page.rect]
    regions = [r for r in _large_images(page)
               if len(page.get_text(clip=r).strip()) < 20 and looks_like_table_image(page, r)]
    return ("table", regions) if regions else ("", [])


OCRLine = tuple[tuple[float, float, float, float], str]  # (x0, y0, x1, y1) on the page, text


def recognize(page: pymupdf.Page, rect: pymupdf.Rect, dpi: int = 300) -> list[OCRLine]:
    """Read the text lines in one area of the page (page coordinates)."""
    img, pix = _pixmap_array(page, rect, dpi)
    result = _get_engine()(img)
    if result.boxes is None or result.txts is None:
        return []
    sx, sy = rect.width / pix.w, rect.height / pix.h
    lines: list[OCRLine] = []
    for box, text, score in zip(result.boxes, result.txts, result.scores, strict=False):
        text = text.strip()
        if not text or score < MIN_CONFIDENCE:
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        lines.append(((rect.x0 + min(xs) * sx, rect.y0 + min(ys) * sy,
                       rect.x0 + max(xs) * sx, rect.y0 + max(ys) * sy), text))
    return lines


def insert_lines(page: pymupdf.Page, lines: list[OCRLine], visible: bool = False) -> int:
    """Write recognised lines onto the page as a text layer, each stretched to its box."""
    if not lines:
        return 0
    page.insert_font(fontname=_FONTNAME, fontbuffer=_FONT.buffer)
    for bbox, text in lines:
        line = pymupdf.Rect(bbox)
        size = max(line.height * 0.8, 1.0)
        width = _FONT.text_length(text, size)
        stretch = pymupdf.Matrix(line.width / width, 1) if width > 0 else pymupdf.Identity
        origin = pymupdf.Point(line.x0, line.y1 - 0.2 * line.height)
        page.insert_text(origin, text, fontsize=size, fontname=_FONTNAME,
                         render_mode=0 if visible else 3,  # 3 = invisible
                         morph=(origin, stretch))
    return len(lines)


def ocr_region(page: pymupdf.Page, rect: pymupdf.Rect, dpi: int = 300, visible: bool = False) -> int:
    """OCR one area of the page and add its text there. Returns the number of lines."""
    return insert_lines(page, recognize(page, rect, dpi), visible)


def page_lines(page: pymupdf.Page, dpi: int = 300) -> list[OCRLine]:
    """All OCR lines this page needs (empty if it has a normal text layer)."""
    _, regions = plan(page)
    return [line for r in regions for line in recognize(page, r, dpi)]


def apply(page: pymupdf.Page, mode: str = "auto", dpi: int = 300, visible: bool = False) -> str:
    """OCR what this page needs. Returns "page", "table" or "" (nothing done).

    ``visible=True`` is used during ingestion (an in-memory copy that is never
    saved): the layout analysis only recognises a table made of *visible* text
    on top of an image; invisible text there is treated as part of the picture.
    Rendering uses invisible text so the page image isn't drawn twice."""
    if mode == "off" or not engine_available():
        return ""
    kind, regions = plan(page)
    lines = sum(ocr_region(page, r, dpi, visible=visible) for r in regions)
    if lines:
        log.debug("OCR page %d (%s): %d lines", page.number + 1, kind, lines)
    return kind if lines else ""
