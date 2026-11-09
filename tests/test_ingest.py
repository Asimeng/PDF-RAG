from ragapp.ingest.chunker import chunk_document, estimate_tokens, split_blocks
from ragapp.ingest.parser import PDFParseError, clean_text, parse_pdf, strip_repeated_lines
from ragapp.schemas import PageText, ParsedDocument

import pytest


def test_strip_repeated_headers_and_page_numbers():
    topics = ["apples", "rivers", "stars", "engines", "music", "forests"]
    words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
    bodies = ["\n".join(f"The {w} line about {t}" for w in words) for t in topics]
    pages = [f"Cell 182, 2020 {1418 + i}\n{b}\n{i + 1}" for i, b in enumerate(bodies)]
    pages[1] = f"1419 Cell 182, 2020\n{bodies[1]}\n2"  # page number on the other side
    assert strip_repeated_lines(pages) == bodies


def test_clean_text_removes_citation_superscripts():
    assert clean_text("similar to reports<sup>14,15</sup> .") == "similar to reports ."
    assert clean_text("shown before<sup>3–5</sup> and<sup>*</sup>") == "shown before and"


def test_clean_text_keeps_meaningful_superscripts():
    # Marker levels and charges change the meaning in biomedical text.
    assert clean_text("HLA-DR<sup>lo</sup> S100A<sup>hi</sup> monocytes") == "HLA-DRlo S100Ahi monocytes"
    assert clean_text("CD14<sup>+</sup> and CD16<sup>−</sup> cells, Ca<sup>2+</sup>") == \
        "CD14+ and CD16− cells, Ca2+"


def test_parse_pdf(sample_pdf):
    doc = parse_pdf(sample_pdf)
    assert doc.num_pages == 4
    assert doc.doc_id and len(doc.doc_id) == 16
    full = " ".join(p.text for p in doc.pages)
    assert "Photosynthesis" in full
    assert "Journal of Tests" not in full  # running header removed


def test_parse_rejects_missing_and_non_pdf(tmp_path):
    with pytest.raises(PDFParseError):
        parse_pdf(tmp_path / "missing.pdf")
    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf")
    with pytest.raises(PDFParseError):
        parse_pdf(bad)


def _doc(pages: list[str]) -> ParsedDocument:
    return ParsedDocument(doc_id="d1", file_name="x.pdf", title="T", num_pages=len(pages),
                          pages=[PageText(page=i + 1, text=t) for i, t in enumerate(pages)])


def test_chunks_respect_size_and_track_pages():
    para = "This is a sentence about topic number {n}. " * 12
    pages = ["# Intro\n\n" + "\n\n".join(para.format(n=i) for i in range(4)),
             "## Methods\n\n" + "\n\n".join(para.format(n=i) for i in range(4, 8))]
    chunks = chunk_document(_doc(pages), chunk_size=200, overlap=30)
    assert len(chunks) > 2
    assert all(estimate_tokens(c.text) <= 260 for c in chunks)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2
    assert chunks[0].section == "Intro" and chunks[-1].section == "Methods"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_references_section_is_skipped():
    pages = ["# Results\n\nThe drug reduced symptoms in most patients over time.",
             "# References\n\n1. Smith J. Some paper. 2001.\n\n2. Doe A. Another. 2002."]
    text = " ".join(c.text for c in chunk_document(_doc(pages), chunk_size=200, overlap=20))
    assert "reduced symptoms" in text and "Smith J" not in text


@pytest.mark.parametrize("text", [
    "6. Williamson, E.J., Walker, A.J., Bhaskaran, K., et al. (2020). Factors associated with "
    "COVID-19-related death using OpenSAFELY. Nature 584, 430–436.",
    "Kamp, V.M., Pillay, J., Lammers, J.-W.J., Pickkers, P., Ulfman, L.H., and Koenderman, L. (2012). "
    "Human suppressive neutrophils exhibit decreased adhesion. J. Leukoc. Biol. 92, 1011–1020.\n\n"
    "Kanehisa, M. (2019). Toward understanding the origin of cellular organisms. Protein Sci. 28, 1947.",
])
def test_detects_reference_entries(text):
    from ragapp.ingest.chunker import looks_like_reference
    assert all(looks_like_reference(p) for p in text.split("\n\n"))


@pytest.mark.parametrize("text", [
    "Clinical presentations of COVID-19 are highly variable (Huang et al., 2020a; Wang et al., 2020).",
    "Smith, the lead author, said results were strong. We repeated the assay in 2020.",
    "PBMCs were isolated from heparinized whole blood by density gradient centrifugation.",
])
def test_normal_text_is_not_a_reference(text):
    from ragapp.ingest.chunker import looks_like_reference
    assert not looks_like_reference(text)


def test_junk_heading_and_picture_text_removed():
    assert "Wilk" not in clean_text("Hi <!-- Start of picture text -->Wilk et al.<!-- End of picture text --> there")
    chunks = chunk_document(_doc(["### **ll**\n\nOPEN text about cells in the blood stream here."]),
                            chunk_size=200, overlap=20)
    assert chunks[0].section == "" and "ll" not in chunks[0].text.split()


def test_long_paragraph_is_split_on_sentences():
    long = " ".join(f"Sentence {i} has several words in it." for i in range(200))
    chunks = chunk_document(_doc([long]), chunk_size=100, overlap=10)
    assert len(chunks) > 5
    assert all(estimate_tokens(c.text) <= 130 for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_document(_doc(["x"]), chunk_size=50, overlap=50)


def test_statement_section_covers_only_next_paragraph():
    text = ("## Abstract\n\nWe studied cells.\n\n###### Data availability\n\nData are on GEO.\n\n"
            "Main text without any heading.\n\n## Methods\n\nWe sequenced.")
    doc = ParsedDocument(doc_id="d", file_name="f.pdf", title="T", num_pages=1,
                         pages=[PageText(page=1, text=text)])
    sections = {b.text: b.section for b in split_blocks(doc) if not b.is_heading}
    assert sections["We studied cells."] == "Abstract"
    assert sections["Data are on GEO."] == "Data availability"
    assert sections["Main text without any heading."] == ""
    assert sections["We sequenced."] == "Methods"
