"""RAGPipeline - the single entry point used by the CLI, the API and the GUI.

    ingest:   PDF -> parse -> chunk -> embed (dense + BM25) -> Qdrant
    ask:      question -> (rewrite with chat history) -> hybrid search
              -> rerank -> prompt with numbered sources -> LLM -> cited answer
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from ragapp.config import RuntimeConfig, resolve_config
from ragapp.generate.llm import LLMError, OllamaLLM
from ragapp.generate.prompts import (
    CONDENSE_PROMPT, NOT_FOUND, build_messages, cited_refs, format_history, format_sources,
    normalize_citations,
)
from ragapp.hardware import is_cloud_model, list_ollama_models, normalize_model_name, use_cuda
from ragapp.index.embeddings import Embedder, SparseEncoder, build_embedder
from ragapp.index.registry import Registry
from ragapp.index.store import VectorStore
from ragapp.ingest.chunker import chunk_document
from ragapp.ingest.parser import PDFParseError, file_sha256, parse_pdf
from ragapp.retrieve.reranker import Reranker
from ragapp.retrieve.retriever import Retriever
from ragapp.schemas import Answer, DocumentInfo, IngestResult, SearchHit, Source

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]  # (message, fraction 0..1)

ANSWER_TOKEN_RESERVE = 1200  # room left in the context window for the answer + prompt


class RAGPipeline:
    def __init__(self, cfg: RuntimeConfig | None = None, *, embedder: Embedder | None = None,
                 llm: OllamaLLM | None = None, reranker: Reranker | None | bool = None):
        self.cfg = cfg or resolve_config()
        s = self.cfg.settings
        log.info("Profile: %s (%s)", self.cfg.profile, self.cfg.profile_reason)
        self.store = VectorStore(s)
        self.registry = Registry(s.registry_path)
        self._embedder = embedder
        self._llm = llm
        self._reranker = reranker  # None = lazy-load, False = disabled
        self._sparse: SparseEncoder | None = None
        self._retriever: Retriever | None = None

    # --------------------------------------------------------- lazy components
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.cfg)
        return self._embedder

    @property
    def sparse(self) -> SparseEncoder:
        if self._sparse is None:
            self._sparse = SparseEncoder(str(self.cfg.settings.model_cache_dir))
        return self._sparse

    @property
    def reranker(self) -> Reranker | None:
        if self._reranker is None:
            name = self.cfg.reranker_model
            if not name or name.lower() == "none":
                self._reranker = False
            else:
                s = self.cfg.settings
                self._reranker = Reranker(name, str(s.model_cache_dir), s.threads, use_cuda(s))
        return self._reranker or None

    @property
    def llm(self) -> OllamaLLM:
        if self._llm is None:
            s, c = self.cfg.settings, self.cfg
            if is_cloud_model(c.llm_model) and not s.allow_cloud_models:
                raise LLMError(f"'{c.llm_model}' is an Ollama cloud model, so your documents would be "
                               "sent to ollama.com. Set RAG_ALLOW_CLOUD_MODELS=true to allow this.")
            self._llm = OllamaLLM(c.llm_model, s.ollama_url, s.ollama_timeout, c.llm_num_ctx,
                                  c.llm_temperature, c.llm_think, s.ollama_keep_alive)
        return self._llm

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.cfg, self.store, self.embedder, self.sparse, self.reranker)
        return self._retriever

    @property
    def collection(self) -> str:
        return self.cfg.collection_name

    def set_llm_model(self, model: str) -> None:
        """Switch the answering model at runtime (no re-indexing required)."""
        self.cfg.llm_model = model
        self._llm = None

    # --------------------------------------------------------------- ingestion
    def ingest_file(self, path: Path, *, force: bool = False, copy_to_library: bool = True,
                    progress: ProgressFn | None = None) -> IngestResult:
        path = Path(path)
        t0 = time.perf_counter()
        report = progress or (lambda msg, frac: None)
        try:
            if path.suffix.lower() != ".pdf":
                raise PDFParseError("Not a PDF file")
            sha = file_sha256(path)
            doc_id = sha[:16]
            if not force and self.registry.get(self.collection, doc_id):
                return IngestResult(file_name=path.name, status="skipped", doc_id=doc_id,
                                    message="already indexed (unchanged)")
            report(f"Reading {path.name}", 0.05)
            doc = parse_pdf(path, ocr_mode=self.cfg.settings.ocr, ocr_dpi=self.cfg.settings.ocr_dpi)
            report(f"Splitting {path.name} into chunks", 0.35)
            s = self.cfg.settings
            chunks = chunk_document(doc, s.chunk_size_tokens, s.chunk_overlap_tokens)
            if not chunks:
                raise PDFParseError("No text chunks produced")
            report(f"Embedding {len(chunks)} chunks", 0.45)
            texts = [c.embed_text for c in chunks]
            dense = self.embedder.embed_documents(texts)
            sparse = self.sparse.embed_documents(texts)
            report("Saving to index", 0.9)
            self.store.ensure_collection(self.collection, len(dense[0]))
            # Same file name but new content = a new version: replace the old one.
            for old in self.registry.find_by_name(self.collection, path.name):
                if old.doc_id != doc_id:
                    self.delete_document(old.doc_id)
            self.store.delete_document(self.collection, doc_id)
            self.store.upsert(self.collection, chunks, dense, sparse)
            self.registry.add(self.collection, doc_id=doc_id, file_name=doc.file_name,
                              title=doc.title, num_pages=doc.num_pages,
                              num_chunks=len(chunks), sha256=sha)
            if copy_to_library:
                self._copy_to_library(path)
            report("Done", 1.0)
            secs = time.perf_counter() - t0
            log.info("Indexed %s: %d chunks in %.1fs", path.name, len(chunks), secs)
            notes = []
            if doc.ocr_pages:
                n = len(doc.ocr_pages)
                notes.append(f"{n} page{'s' * (n != 1)} read with OCR")
            if doc.unread_scanned_pages:
                notes.append(f"{doc.unread_scanned_pages} scanned page(s) skipped: "
                             "install the OCR engine (pip install rapidocr)")
            return IngestResult(file_name=path.name, status="indexed", doc_id=doc_id,
                                num_chunks=len(chunks), seconds=round(secs, 1),
                                message="; ".join(notes))
        except PDFParseError as exc:
            log.warning("Skipping %s: %s", path.name, exc)
            return IngestResult(file_name=path.name, status="failed", message=str(exc),
                                seconds=round(time.perf_counter() - t0, 1))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a batch
            # Nothing is written to the index before embedding succeeds, so an older
            # version of the document (if any) is still intact.
            log.exception("Failed to index %s", path.name)
            return IngestResult(file_name=path.name, status="failed",
                                message=f"Unexpected error: {exc}",
                                seconds=round(time.perf_counter() - t0, 1))

    def ingest_path(self, path: Path, *, force: bool = False,
                    progress: ProgressFn | None = None) -> list[IngestResult]:
        path = Path(path)
        if path.is_dir():
            files = sorted(
                f for f in path.rglob("*")
                if f.suffix.lower() == ".pdf"
                and not any(p.startswith(".") or p in {"site-packages", "node_modules", "__pycache__"}
                            for p in f.relative_to(path).parts[:-1])
            )
        else:
            files = [path]
        results = []
        for i, f in enumerate(files):
            def sub(msg: str, frac: float, i=i) -> None:
                if progress:
                    progress(f"[{i + 1}/{len(files)}] {msg}", (i + frac) / len(files))
            results.append(self.ingest_file(f, force=force, progress=sub))
        return results

    def _copy_to_library(self, path: Path) -> None:
        lib = self.cfg.settings.pdf_dir
        lib.mkdir(parents=True, exist_ok=True)
        target = lib / path.name
        try:
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
        except OSError as exc:
            log.warning("Could not copy %s to library: %s", path.name, exc)

    def delete_document(self, doc_id: str) -> bool:
        info = self.registry.get(self.collection, doc_id)
        self.store.delete_document(self.collection, doc_id)
        self.registry.remove(self.collection, doc_id)
        return info is not None

    def list_documents(self) -> list[DocumentInfo]:
        return self.registry.list(self.collection)

    # --------------------------------------------------------------- querying
    def search(self, query: str, top_k: int | None = None,
               doc_ids: list[str] | None = None) -> list[SearchHit]:
        return self.retriever.retrieve(query, top_k=top_k, doc_ids=doc_ids)

    def _standalone_question(self, question: str, history: list[dict] | None) -> str:
        if not history:
            return question
        prompt = CONDENSE_PROMPT.format(history=format_history(history), question=question)
        try:
            rewritten = self.llm.chat([{"role": "user", "content": prompt}]).strip().strip('"')
            return rewritten.splitlines()[0] if rewritten else question
        except Exception as exc:  # never fail a question because rewriting failed
            log.warning("Question rewrite failed (%s); using original", exc)
            return question

    def _prepare(self, question: str, history: list[dict] | None, doc_ids: list[str] | None):
        timings: dict[str, float] = {}
        t = time.perf_counter()
        standalone = self._standalone_question(question, history)
        timings["rewrite"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        hits = self.search(standalone, doc_ids=doc_ids)
        timings["retrieve"] = (time.perf_counter() - t) * 1000
        budget = max(500, self.cfg.llm_num_ctx - ANSWER_TOKEN_RESERVE)
        sources_text, used = format_sources(hits, budget)
        sources = [
            Source(ref=i + 1, doc_id=h.chunk.doc_id, file_name=h.chunk.file_name, title=h.chunk.title,
                   page_start=h.chunk.page_start, page_end=h.chunk.page_end,
                   section=h.chunk.section, text=h.chunk.text,
                   score=round(h.rerank_score if h.rerank_score is not None else h.score, 4))
            for i, h in enumerate(used)
        ]
        return standalone, sources_text, sources, timings

    def _check_index(self) -> None:
        if self.store.count(self.collection) == 0:
            others = [c for c in self.registry.collections() if c != self.collection]
            hint = (f" Documents exist in other indexes ({', '.join(others)}) built with a "
                    "different embedding model - re-ingest them for the current profile."
                    if others else " Add PDFs first (rag ingest <folder>).")
            raise RuntimeError(f"No documents are indexed for '{self.collection}'.{hint}")

    def _finish(self, question, standalone, answer_text, sources, timings) -> Answer:
        answer_text = normalize_citations(answer_text)
        refs = cited_refs(answer_text)
        for s in sources:
            s.cited = s.ref in refs
        return Answer(question=question, standalone_question=standalone,
                      answer=answer_text.strip(), sources=sources, model=self.cfg.llm_model,
                      profile=self.cfg.profile,
                      timings_ms={k: round(v, 1) for k, v in timings.items()})

    def ask(self, question: str, history: list[dict] | None = None,
            doc_ids: list[str] | None = None) -> Answer:
        self._check_index()
        standalone, sources_text, sources, timings = self._prepare(question, history, doc_ids)
        if not sources:
            return self._finish(question, standalone, NOT_FOUND, [], timings)
        t = time.perf_counter()
        text = self.llm.chat(build_messages(standalone, sources_text))
        timings["generate"] = (time.perf_counter() - t) * 1000
        return self._finish(question, standalone, text, sources, timings)

    def ask_stream(self, question: str, history: list[dict] | None = None,
                   doc_ids: list[str] | None = None) -> Iterator[dict]:
        """Yields events: {"type": "sources"}, many {"type": "token"}, then {"type": "done"}."""
        self._check_index()
        standalone, sources_text, sources, timings = self._prepare(question, history, doc_ids)
        yield {"type": "sources", "standalone_question": standalone,
               "sources": [s.model_dump() for s in sources]}
        if not sources:
            yield {"type": "token", "text": NOT_FOUND}
            yield {"type": "done", "answer": self._finish(question, standalone, NOT_FOUND, [], timings).model_dump()}
            return
        t = time.perf_counter()
        first_token_at = None
        pieces: list[str] = []
        for piece in self.llm.stream(build_messages(standalone, sources_text)):
            if first_token_at is None:
                first_token_at = time.perf_counter()
                timings["first_token"] = (first_token_at - t) * 1000
            pieces.append(piece)
            yield {"type": "token", "text": piece}
        timings["generate"] = (time.perf_counter() - t) * 1000
        answer = self._finish(question, standalone, "".join(pieces), sources, timings)
        yield {"type": "done", "answer": answer.model_dump()}

    # --------------------------------------------------------------- status
    def status(self) -> dict:
        s = self.cfg.settings
        installed = list_ollama_models(s.ollama_url)
        needed = [self.cfg.llm_model] + ([self.cfg.embed_model] if self.cfg.embed_provider == "ollama" else [])
        missing = ([m for m in needed if normalize_model_name(m) not in set(installed)]
                   if installed is not None else needed)
        return {
            "config": self.cfg.public_dict(),
            "ollama": {"url": s.ollama_url, "reachable": installed is not None,
                       "installed_models": installed or [], "missing_models": missing},
            "index": {"store": self.store.mode, "collection": self.collection,
                      "documents": len(self.list_documents()),
                      "chunks": self.store.count(self.collection),
                      "other_indexes": {k: v for k, v in self.registry.collections().items()
                                        if k != self.collection}},
        }

    def close(self) -> None:
        self.store.close()
        self.registry.close()
