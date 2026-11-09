"""Embedding models: turn text into vectors (lists of numbers) that capture meaning.

Two interchangeable providers:
  * ``fastembed`` - runs ONNX models inside this Python process. No Ollama
    needed; ideal for CPU-only machines.
  * ``ollama``    - asks the Ollama server (e.g. on a GPU box) to embed.

Many embedding models are trained with *instruction prefixes* that tell them
whether a text is a question or a document. Getting these right noticeably
improves retrieval, so they are handled here per model family.
"""

from __future__ import annotations

import logging
from typing import Protocol

from ragapp.config import RuntimeConfig
from ragapp.hardware import use_cuda

log = logging.getLogger(__name__)

# (query prefix, document prefix) per model family (matched on lower-cased name).
PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed": ("search_query: ", "search_document: "),
    "qwen3-embedding": (
        "Instruct: Given a question, retrieve passages from documents that answer the question\nQuery: ",
        "",
    ),
    "mxbai-embed": ("Represent this sentence for searching relevant passages: ", ""),
    "bge-": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
}


CPU_MAX_BATCH = 8


def prefixes_for(model: str) -> tuple[str, str]:
    name = model.lower()
    for key, value in PREFIXES.items():
        if key in name:
            return value
    return ("", "")


class Embedder(Protocol):
    model_name: str

    @property
    def dim(self) -> int: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str, threads: int | None, cuda: bool, batch_size: int):
        from fastembed import TextEmbedding

        self.model_name = model_name
        # On CPU, big batches barely speed things up but multiply peak memory: attention
        # needs ~batch x length^2 memory, and 32 long passages can request >1.5 GB at once.
        self.batch_size = batch_size if cuda else min(batch_size, CPU_MAX_BATCH)
        self._q, self._d = prefixes_for(model_name)
        log.info("Loading embedding model %s...", model_name)
        kwargs = {"cache_dir": cache_dir, "threads": threads}
        if cuda:
            kwargs["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._model = TextEmbedding(model_name, **kwargs)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        docs = [self._d + t for t in texts]
        return [v.tolist() for v in self._model.embed(docs, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([self._q + text]))).tolist()


class OllamaEmbedder:
    def __init__(self, model_name: str, url: str, timeout: float, batch_size: int, keep_alive: str):
        import ollama

        self.model_name = model_name
        self.batch_size = batch_size
        self.keep_alive = keep_alive
        self._q, self._d = prefixes_for(model_name)
        self._client = ollama.Client(host=url, timeout=timeout)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embed(
            model=self.model_name, input=texts, truncate=True, keep_alive=self.keep_alive
        )
        return [list(v) for v in resp["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed([self._d + t for t in texts[i : i + self.batch_size]]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._q + text])[0]


def build_embedder(cfg: RuntimeConfig) -> Embedder:
    s = cfg.settings
    if cfg.embed_provider == "fastembed":
        return FastEmbedEmbedder(
            cfg.embed_model, str(s.model_cache_dir), s.threads, use_cuda(s), s.embed_batch_size
        )
    if cfg.embed_provider == "ollama":
        return OllamaEmbedder(
            cfg.embed_model, s.ollama_url, s.ollama_timeout, s.embed_batch_size, s.ollama_keep_alive
        )
    raise ValueError(f"Unknown embedding provider: {cfg.embed_provider}")


class SparseEncoder:
    """BM25 keyword weights, stored as a sparse vector next to the dense one.

    Qdrant applies the IDF (how rare a word is) at search time, so documents can
    be added incrementally without recomputing statistics."""

    def __init__(self, cache_dir: str):
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding("Qdrant/bm25", cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [(e.indices.tolist(), e.values.tolist()) for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> tuple[list[int], list[float]]:
        e = next(iter(self._model.query_embed(text)))
        return e.indices.tolist(), e.values.tolist()
