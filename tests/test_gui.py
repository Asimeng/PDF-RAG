"""GUI layer: page rendering, the API client, and a headless run of the Streamlit app."""

from pathlib import Path

import pytest

from ragapp.gui.client import APIError, RAGClient
from ragapp.render import matched_word_indices, render_page
from tests.test_api import _upload, client  # noqa: F401  (client is a fixture)

PNG = b"\x89PNG"


# ---------------------------------------------------------------- highlighting
def test_alignment_tolerates_markdown_and_hyphenation():
    page = "Intro text here . Finally , a novel cell popu- lation was increased in ARDS patients . Unrelated words follow".split()
    passage = "**Finally**, a novel cell population was increased in ARDS patients."
    hit = [page[i] for i in matched_word_indices(page, passage)]
    # The hyphen-split word is filled in as a small gap between matched runs.
    assert hit[:4] == ["Finally", ",", "a", "novel"]
    assert "popu-" in hit and "lation" in hit and "patients" in hit
    assert "Intro" not in hit and "Unrelated" not in hit


def test_alignment_ignores_short_coincidences():
    assert matched_word_indices("the cat sat on the mat".split(), "the dog") == []
    assert matched_word_indices(["a"], "") == []


def test_render_page(sample_pdf):
    assert render_page(sample_pdf, 1).startswith(PNG)
    assert render_page(sample_pdf, 1, highlight="Photosynthesis converts light").startswith(PNG)
    with pytest.raises(ValueError):
        render_page(sample_pdf, 99)


# ---------------------------------------------------------------- API client
def test_client_end_to_end(client, sample_pdf):  # noqa: F811
    rc = RAGClient("http://testserver", http=client)
    assert rc.health()
    job = rc.upload([(sample_pdf.name, sample_pdf.read_bytes())])
    while job["status"] in {"queued", "running"}:
        job = rc.job(job["job_id"])
    assert job["results"][0]["status"] == "indexed"

    doc = rc.documents()[0]
    assert rc.document_file(doc["doc_id"]).startswith(b"%PDF")
    assert rc.page_image(doc["doc_id"], 1, "Photosynthesis converts light").startswith(PNG)

    events = list(rc.ask_stream("What is photosynthesis?"))
    assert [e["type"] for e in events][0] == "sources" and events[-1]["type"] == "done"
    answer = events[-1]["answer"]
    assert answer["sources"][0]["doc_id"] == doc["doc_id"]
    assert any(s["cited"] for s in answer["sources"])

    with pytest.raises(APIError) as err:
        rc.page_image(doc["doc_id"], 99)
    assert err.value.status == 404
    rc.delete_document(doc["doc_id"])
    assert rc.documents() == []


def test_client_unreachable_message():
    rc = RAGClient("http://127.0.0.1:9", timeout=2)
    assert rc.health() is False
    with pytest.raises(APIError, match="Can't reach"):
        rc.documents()


# ---------------------------------------------------------------- Streamlit app
APP = str(Path(__file__).resolve().parents[1] / "src" / "ragapp" / "gui" / "app.py")
DOC = {"doc_id": "d1", "file_name": "plants.pdf", "title": "Plant Biology Basics", "num_pages": 2,
       "num_chunks": 3, "sha256": "x", "ingested_at": "2026-11-08T10:00:00", "collection": "c"}
SOURCE = {"ref": 1, "doc_id": "d1", "file_name": "plants.pdf", "title": "Plant Biology Basics",
          "page_start": 1, "page_end": 1, "section": "Intro", "text": "Photosynthesis converts light.",
          "score": 0.9, "cited": True}


@pytest.fixture
def fake_api(monkeypatch):
    answer = {"answer": "It converts light into energy [1].", "sources": [SOURCE], "model": "fake",
              "standalone_question": "What is photosynthesis?"}
    fakes = {
        "health": lambda self: True,
        "documents": lambda self: [DOC],
        "page_image": lambda self, *a, **k: render_page_png(),
        "document_file": lambda self, doc_id: b"%PDF-1.7",
        "ask_stream": lambda self, q, h=None, d=None: iter([
            {"type": "sources", "sources": [SOURCE]},
            {"type": "token", "text": answer["answer"]},
            {"type": "done", "answer": answer},
        ]),
    }
    for name, fn in fakes.items():
        monkeypatch.setattr(RAGClient, name, fn)


def render_page_png() -> bytes:
    import pymupdf

    return pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), 0).tobytes("png")


def test_gui_asks_and_shows_sources(fake_api):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Ask your documents"

    at.chat_input[0].set_value("What is photosynthesis?").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert ":blue-badge[1]" in text  # citation rendered as a badge
    assert any("Sources: 1 cited" in e.label for e in at.expander)


def test_gui_unreachable_api(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(RAGClient, "health", lambda self: False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "Can't reach the service" in at.error[0].value
