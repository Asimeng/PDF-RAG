import json
import time

import pytest
from fastapi.testclient import TestClient

from ragapp import api
from ragapp.config import PROFILES, build_runtime_config
from tests.conftest import FakeEmbedder, FakeLLM


@pytest.fixture
def client(settings, monkeypatch):
    from ragapp.pipeline import RAGPipeline

    cfg = build_runtime_config(settings, PROFILES["cpu"], "test")
    monkeypatch.setattr(api, "RAGPipeline",
                        lambda: RAGPipeline(cfg, embedder=FakeEmbedder(), llm=FakeLLM(), reranker=False))
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    with TestClient(api.app) as c:
        yield c


def _upload(client, pdf_path):
    with open(pdf_path, "rb") as f:
        r = client.post("/documents", files=[("files", (pdf_path.name, f, "application/pdf"))])
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.1)
    raise AssertionError("ingestion job did not finish")


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_upload_ask_stream_delete(client, sample_pdf):
    job = _upload(client, sample_pdf)
    assert job["status"] == "done" and job["results"][0]["status"] == "indexed"

    docs = client.get("/documents").json()
    assert len(docs) == 1
    doc_id = docs[0]["doc_id"]
    assert client.get(f"/documents/{doc_id}/file").headers["content-type"] == "application/pdf"

    r = client.post("/ask", json={"question": "What is photosynthesis?"})
    assert r.status_code == 200 and r.json()["sources"]

    with client.stream("POST", "/ask/stream", json={"question": "What is chlorophyll?"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"

    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert client.get("/documents").json() == []


def test_rejects_non_pdf(client):
    r = client.post("/documents", files=[("files", ("evil.pdf", b"MZ not a pdf", "application/pdf"))])
    assert r.status_code == 415


def test_ask_on_empty_index_is_409(client):
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 409 and "No documents" in r.json()["detail"]


def test_api_key(client, settings):
    settings.api_key = "s3cret"
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for probes


def test_cloud_model_switch_is_blocked(client):
    r = client.put("/config/llm", json={"model": "glm-5.3:cloud"})
    assert r.status_code == 403 and "ollama.com" in r.json()["detail"]
