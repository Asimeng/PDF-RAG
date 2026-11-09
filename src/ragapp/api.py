"""REST API (FastAPI).  Run:  rag serve   ->  interactive docs at /docs

Endpoints
  GET    /health                      liveness probe
  GET    /status                      profile, models, Ollama, index stats
  GET    /profiles                    available hardware profiles
  GET    /models                      models installed on Ollama
  PUT    /config/llm                  switch the answering model at runtime
  GET    /documents                   list indexed documents
  POST   /documents                   upload + index PDFs (background job)
  GET    /jobs/{job_id}               progress of an upload job
  GET    /documents/{doc_id}/file     download the original PDF
  POST   /documents/{doc_id}/pages/{n}/image   page as PNG, passage highlighted
  DELETE /documents/{doc_id}          remove a document
  POST   /search                      retrieval only (debugging)
  POST   /ask                         question -> cited answer (JSON)
  POST   /ask/stream                  same, streamed as Server-Sent Events
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from ragapp import __version__
from ragapp.config import PROFILES, get_settings
from ragapp.generate.llm import LLMError
from ragapp.hardware import is_cloud_model, list_ollama_models
from ragapp.logging_setup import setup_logging
from ragapp.pipeline import RAGPipeline
from ragapp.render import render_page
from ragapp.schemas import Answer, DocumentInfo

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- app state
class State:
    pipeline: RAGPipeline
    executor: ThreadPoolExecutor  # one worker => ingestion jobs run one at a time
    jobs: dict[str, dict]
    jobs_lock: threading.Lock


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    state.pipeline = RAGPipeline()
    state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
    state.jobs = {}
    state.jobs_lock = threading.Lock()
    log.info("API ready - profile %s, index %s", state.pipeline.cfg.profile, state.pipeline.collection)
    yield
    state.executor.shutdown(wait=False, cancel_futures=True)
    state.pipeline.close()


app = FastAPI(title="PDF RAG API", version=__version__, lifespan=lifespan,
              description="Ask questions about your PDF documents. Answers cite file and page.")
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins,
                   allow_methods=["*"], allow_headers=["*"])

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(_api_key_header)) -> None:
    expected = get_settings().api_key
    if expected and key != expected:
        raise HTTPException(401, "Invalid or missing X-API-Key header")


auth = [Depends(require_key)]


@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_: Request, exc: RuntimeError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# ----------------------------------------------------------------- models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    history: list[dict] | None = Field(None, description='[{"role": "user"|"assistant", "content": "..."}]')
    doc_ids: list[str] | None = Field(None, description="Restrict the search to these documents")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(5, ge=1, le=50)
    doc_ids: list[str] | None = None


class LLMSwitch(BaseModel):
    model: str


# ----------------------------------------------------------------- system
@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/status", dependencies=auth)
def status():
    return state.pipeline.status()


@app.get("/profiles", dependencies=auth)
def profiles():
    return [asdict(p) for p in PROFILES.values()]


@app.get("/models", dependencies=auth)
def models():
    installed = list_ollama_models(get_settings().ollama_url)
    if installed is None:
        raise HTTPException(503, "Ollama is not reachable")
    allow_cloud = get_settings().allow_cloud_models
    return {"models": [m for m in installed if allow_cloud or not is_cloud_model(m)],
            "active": state.pipeline.cfg.llm_model}


@app.put("/config/llm", dependencies=auth)
def switch_llm(body: LLMSwitch):
    if is_cloud_model(body.model) and not get_settings().allow_cloud_models:
        raise HTTPException(403, f"'{body.model}' is a cloud model: your documents would be sent to "
                                 "ollama.com. Set RAG_ALLOW_CLOUD_MODELS=true to allow this.")
    installed = list_ollama_models(get_settings().ollama_url) or []
    name = body.model if ":" in body.model else f"{body.model}:latest"
    if installed and name not in installed:
        raise HTTPException(404, f"Model '{body.model}' is not installed on Ollama")
    state.pipeline.set_llm_model(body.model)
    return {"active": body.model}


# ----------------------------------------------------------------- documents
@app.get("/documents", response_model=list[DocumentInfo], dependencies=auth)
def list_documents():
    return state.pipeline.list_documents()


def _safe_name(name: str) -> str:
    name = Path(name or "upload.pdf").name
    name = re.sub(r"[^\w.\- ()]+", "_", name).strip() or "upload.pdf"
    return name if name.lower().endswith(".pdf") else name + ".pdf"


@app.post("/documents", status_code=202, dependencies=auth)
async def upload_documents(files: list[UploadFile] = File(...)):
    settings = get_settings()
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    limit = settings.max_upload_mb * 1024 * 1024
    saved: list[Path] = []
    for f in files:
        data = await f.read()
        if len(data) > limit:
            raise HTTPException(413, f"{f.filename} exceeds {settings.max_upload_mb} MB")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, f"{f.filename} is not a PDF")
        target = settings.pdf_dir / _safe_name(f.filename)
        target.write_bytes(data)
        saved.append(target)

    job_id = uuid.uuid4().hex[:12]
    job = {"job_id": job_id, "status": "queued", "progress": 0.0, "message": "Queued",
           "files": [p.name for p in saved], "results": []}
    with state.jobs_lock:
        state.jobs[job_id] = job

    def run():
        job["status"] = "running"
        try:
            for i, path in enumerate(saved):
                def prog(msg: str, frac: float, i=i) -> None:
                    job["message"] = f"[{i + 1}/{len(saved)}] {msg}"
                    job["progress"] = round((i + frac) / len(saved), 3)
                res = state.pipeline.ingest_file(path, copy_to_library=False, progress=prog)
                job["results"].append(res.model_dump())
            job["status"], job["progress"], job["message"] = "done", 1.0, "Finished"
        except Exception as exc:  # surface unexpected failures to the client
            log.exception("Ingestion job %s failed", job_id)
            job["status"], job["message"] = "failed", str(exc)

    state.executor.submit(run)
    return job


@app.get("/jobs/{job_id}", dependencies=auth)
def get_job(job_id: str):
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job


@app.get("/documents/{doc_id}/file", dependencies=auth)
def get_document_file(doc_id: str):
    info = state.pipeline.registry.get(state.pipeline.collection, doc_id)
    if not info:
        raise HTTPException(404, "Unknown document")
    path = get_settings().pdf_dir / info.file_name
    if not path.exists():
        raise HTTPException(404, "Original PDF is no longer in the library folder")
    return FileResponse(path, media_type="application/pdf", filename=info.file_name)


class RenderRequest(BaseModel):
    highlight: str | None = Field(None, max_length=20000, description="Passage to highlight")
    zoom: float = Field(1.6, ge=0.5, le=4.0)


@app.post("/documents/{doc_id}/pages/{page}/image", dependencies=auth,
          responses={200: {"content": {"image/png": {}}}})
def render_document_page(doc_id: str, page: int, body: RenderRequest | None = None):
    """PNG of one page, with the given passage highlighted (used to show citations)."""
    info = state.pipeline.registry.get(state.pipeline.collection, doc_id)
    if not info:
        raise HTTPException(404, "Unknown document")
    path = get_settings().pdf_dir / info.file_name
    if not path.exists():
        raise HTTPException(404, "Original PDF is no longer in the library folder")
    body = body or RenderRequest()
    try:
        png = render_page(path, page, body.highlight, body.zoom, ocr_mode=get_settings().ocr)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/documents/{doc_id}", dependencies=auth)
def delete_document(doc_id: str, delete_file: bool = False):
    info = state.pipeline.registry.get(state.pipeline.collection, doc_id)
    if not info:
        raise HTTPException(404, "Unknown document")
    state.pipeline.delete_document(doc_id)
    if delete_file:
        (get_settings().pdf_dir / info.file_name).unlink(missing_ok=True)
    return {"deleted": doc_id}


# ----------------------------------------------------------------- querying
@app.post("/search", dependencies=auth)
def search(body: SearchRequest):
    hits = state.pipeline.search(body.query, top_k=body.top_k, doc_ids=body.doc_ids)
    return [h.model_dump() for h in hits]


@app.post("/ask", response_model=Answer, dependencies=auth)
def ask(body: AskRequest):
    return state.pipeline.ask(body.question, history=body.history, doc_ids=body.doc_ids)


@app.post("/ask/stream", dependencies=auth)
def ask_stream(body: AskRequest):
    """Server-Sent Events: each line is `data: {json}` with type sources|token|done|error."""
    def events():
        try:
            for ev in state.pipeline.ask_stream(body.question, history=body.history, doc_ids=body.doc_ids):
                yield f"data: {json.dumps(ev)}\n\n"
        except (LLMError, RuntimeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
