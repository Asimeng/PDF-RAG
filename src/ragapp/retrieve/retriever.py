"""Retriever: question -> the most relevant chunks."""

from __future__ import annotations

import logging

from ragapp.config import RuntimeConfig
from ragapp.index.embeddings import Embedder, SparseEncoder
from ragapp.index.store import VectorStore
from ragapp.retrieve.reranker import Reranker
from ragapp.schemas import SearchHit

log = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        cfg: RuntimeConfig,
        store: VectorStore,
        embedder: Embedder,
        sparse: SparseEncoder,
        reranker: Reranker | None,
    ):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self.sparse = sparse
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> list[SearchHit]:
        """mode: hybrid | dense | sparse.  rerank=False skips the cross-encoder."""
        top_k = top_k or self.cfg.top_k_context
        candidates = self.store.hybrid_search(
            self.cfg.collection_name,
            self.embedder.embed_query(query),
            self.sparse.embed_query(query),
            limit=max(self.cfg.top_k_retrieve, top_k),
            doc_ids=doc_ids,
            mode=mode,
        )
        if self.reranker is None or not rerank:
            return candidates[:top_k]
        hits = self.reranker.rerank(query, candidates, top_k)
        if self.cfg.min_rerank_score is not None:
            kept = [h for h in hits if (h.rerank_score or 0) >= self.cfg.min_rerank_score]
            log.debug("min_rerank_score kept %d/%d", len(kept), len(hits))
            hits = kept
        return hits
