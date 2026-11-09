"""Cross-encoder reranking.

Search (embeddings + BM25) is fast but approximate: it compares a question and
a passage *separately encoded*. A cross-encoder reads the question and passage
*together* and scores how well the passage answers it - slower, but far more
accurate. So we search broadly (e.g. 20-50 candidates) and let the reranker
pick the best few.
"""

from __future__ import annotations

import logging

from ragapp.schemas import SearchHit

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, cache_dir: str, threads: int | None, cuda: bool):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        log.info("Loading reranker %s...", model_name)
        kwargs = {"cache_dir": cache_dir, "threads": threads}
        if cuda:
            kwargs["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._model = TextCrossEncoder(model_name, **kwargs)

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        docs = [h.chunk.embed_text for h in hits]
        scores = list(self._model.rerank(query, docs, batch_size=16))
        for h, s in zip(hits, scores, strict=True):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_k]
