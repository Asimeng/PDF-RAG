"""OCR for scanned pages and tables saved as images (needs the rapidocr package)."""

from pathlib import Path

import pymupdf
import pytest

pytest.importorskip("rapidocr")

from ragapp.ingest import ocr  # noqa: E402
from ragapp.ingest.parser import PDFParseError, parse_pdf  # noqa: E402
from ragapp.render import render_page  # noqa: E402

BODY = ("We enrolled 412 adults in a randomized trial of a high-dose influenza vaccine. "
        "Vaccine efficacy was 24.2 percent compared with the standard dose.")


def _as_image(page: pymupdf.Page, dpi: int = 200) -> pymupdf.Pixmap:
    return page.get_pixmap(dpi=dpi)


def make_scanned_pdf(path: Path) -> Path:
    """A page that is only a picture of text, like a scanner produces."""
    src = pymupdf.open()
    pg = src.new_page(width=420, height=220)
    pg.insert_text((20, 40), "Influenza Vaccination Study", fontsize=16)
    pg.insert_textbox(pymupdf.Rect(20, 60, 400, 200), BODY, fontsize=11)
    out = pymupdf.open()
    page = out.new_page(width=420, height=220)
    page.insert_image(page.rect, pixmap=_as_image(pg))
    out.save(path)
    return path


def _table_pixmap() -> pymupdf.Pixmap:
    t = pymupdf.open()
    tp = t.new_page(width=500, height=170)
    rows = [("Group", "Patients", "Cases"), ("High dose", "206", "31"), ("Standard", "206", "41")]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            tp.insert_text((20 + j * 150, 40 + i * 40), cell, fontsize=16)
        tp.draw_line((15, 50 + i * 40), (480, 50 + i * 40))
    return _as_image(tp)


def _chart_pixmap() -> pymupdf.Pixmap:
    """A coloured bar chart with gridlines: a figure, not a table."""
    c = pymupdf.open()
    cp = c.new_page(width=500, height=200)
    for y in (40, 80, 120, 160):
        cp.draw_line((20, y), (480, y), color=(0.7, 0.7, 0.7))
    for i, h in enumerate((60, 110, 90, 140)):
        cp.draw_rect(pymupdf.Rect(60 + i * 100, 170 - h, 120 + i * 100, 170),
                     color=None, fill=[(0.9, 0.2, 0.2), (0.2, 0.5, 0.9), (0.2, 0.7, 0.3), (0.9, 0.6, 0.1)][i])
    cp.insert_text((60, 190), "IL-6   TNF   IL-1   IFN", fontsize=12)
    return _as_image(cp)


def make_mixed_pdf(path: Path, image: pymupdf.Pixmap) -> Path:
    """Normal text plus one image (a table or a chart)."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 520, 200),
                        "Results. Table 1 shows influenza cases by vaccine group. " * 3, fontsize=11)
    page.insert_image(pymupdf.Rect(72, 220, 520, 380), pixmap=image)
    doc.save(path)
    return path


def test_scanned_page_is_read(tmp_path):
    doc = parse_pdf(make_scanned_pdf(tmp_path / "scan.pdf"))
    text = doc.pages[0].text
    assert "412 adults" in text and "24.2 percent" in text
    assert doc.ocr_pages == {1: "page"}
    assert doc.title == "Influenza Vaccination Study"  # title found in the OCR'd text


def test_table_image_becomes_a_table(tmp_path):
    doc = parse_pdf(make_mixed_pdf(tmp_path / "table.pdf", _table_pixmap()))
    text = doc.pages[0].text
    assert "|High dose|206|31|" in text.replace(" |", "|").replace("| ", "|")
    assert doc.ocr_pages == {1: "table"}


def test_figures_are_not_ocrd(tmp_path):
    doc = parse_pdf(make_mixed_pdf(tmp_path / "chart.pdf", _chart_pixmap()))
    assert doc.ocr_pages == {}
    assert "IL-6" not in doc.pages[0].text


def test_normal_text_pdf_needs_no_ocr(sample_pdf):
    assert parse_pdf(sample_pdf).ocr_pages == {}


def test_ocr_off_and_missing_engine_give_clear_errors(tmp_path, monkeypatch):
    scan = make_scanned_pdf(tmp_path / "scan.pdf")
    with pytest.raises(PDFParseError, match="RAG_OCR=off"):
        parse_pdf(scan, ocr_mode="off")
    monkeypatch.setattr(ocr, "engine_available", lambda: False)
    with pytest.raises(PDFParseError, match="pip install rapidocr"):
        parse_pdf(scan)


def test_highlight_works_on_scanned_page(tmp_path):
    scan = make_scanned_pdf(tmp_path / "scan.pdf")
    plain = render_page(scan, 1, zoom=1.0)
    marked = render_page(scan, 1, highlight=BODY, zoom=1.0)
    assert marked.startswith(b"\x89PNG") and marked != plain  # yellow highlight was drawn
    # The invisible OCR layer must not change the page image itself.
    assert render_page(scan, 1, highlight="words that are not on the page at all", zoom=1.0) == plain
