import pytest

from ragapp.generate.prompts import NOT_FOUND


def test_ingest_search_ask_delete(pipeline, sample_pdf):
    res = pipeline.ingest_file(sample_pdf)
    assert res.status == "indexed" and res.num_chunks > 0

    # Unchanged file is skipped the second time.
    assert pipeline.ingest_file(sample_pdf).status == "skipped"

    docs = pipeline.list_documents()
    assert len(docs) == 1 and docs[0].file_name == "biology.pdf"

    hits = pipeline.search("Which enzyme fixes carbon dioxide in the Calvin cycle?", top_k=2)
    assert any("RuBisCO" in h.chunk.text for h in hits)

    answer = pipeline.ask("What is photosynthesis?")
    assert answer.sources and answer.sources[0].cited  # FakeLLM cites [1]
    assert "Sources:" in pipeline.llm.calls[-1][1]["content"]

    assert pipeline.delete_document(res.doc_id)
    assert pipeline.list_documents() == []
    assert pipeline.store.count(pipeline.collection) == 0


def test_stream_events(pipeline, sample_pdf):
    pipeline.ingest_file(sample_pdf)
    events = list(pipeline.ask_stream("What is chlorophyll?"))
    assert events[0]["type"] == "sources"
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["type"] == "done" and events[-1]["answer"]["answer"]


def test_ask_without_documents_gives_clear_error(pipeline):
    with pytest.raises(RuntimeError, match="No documents are indexed"):
        pipeline.ask("anything")


def test_failed_pdf_is_reported_not_raised(pipeline, tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 garbage")
    res = pipeline.ingest_file(bad)
    assert res.status == "failed" and res.message


def test_new_version_replaces_old(pipeline, sample_pdf, tmp_path):
    from tests.conftest import make_pdf

    first = pipeline.ingest_file(sample_pdf)
    make_pdf(sample_pdf, ["Completely new content about volcanoes and magma chambers."])
    second = pipeline.ingest_file(sample_pdf)
    assert second.status == "indexed" and second.doc_id != first.doc_id
    assert [d.doc_id for d in pipeline.list_documents()] == [second.doc_id]


def test_history_triggers_question_rewrite(pipeline, sample_pdf):
    pipeline.ingest_file(sample_pdf)
    pipeline.llm.reply = "What does RuBisCO do?"
    ans = pipeline.ask("What does it do?", history=[
        {"role": "user", "content": "Tell me about RuBisCO"},
        {"role": "assistant", "content": "RuBisCO is an enzyme [1]."}])
    assert ans.standalone_question == "What does RuBisCO do?"


def test_no_hits_returns_not_found(pipeline, sample_pdf):
    pipeline.ingest_file(sample_pdf)
    ans = pipeline.ask("Photosynthesis?", doc_ids=["does-not-exist"])
    assert ans.answer == NOT_FOUND and ans.sources == []
