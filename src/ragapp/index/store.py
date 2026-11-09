"""Qdrant vector store with hybrid (dense + BM25) search.

Embedded mode (default) stores everything in ``data/qdrant`` - no server needed.
Set ``RAG_QDRANT_URL`` (e.g. ``http://qdrant:6333``) to use a Qdrant server
instead - same code, better for multi-user deployments.

Note: in embedded mode only ONE process may open the storage folder at a time
(e.g. run either the API server or the CLI, not both simultaneously).
"""

from __future__ import annotations

import logging
import threading

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from ragapp.config import Settings
from ragapp.schemas import Chunk, SearchHit

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"


class IndexMismatchError(Exception):
    pass


class VectorStore:
    def __init__(self, settings: Settings):
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            self.mode = f"server ({settings.qdrant_url})"
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            try:
                self.client = QdrantClient(path=str(settings.qdrant_path))
            except RuntimeError as exc:
                if "already accessed" in str(exc):
                    raise RuntimeError(
                        "The index folder is in use by another process (probably the API server). "
                        "Stop it first, or use the API instead of the CLI."
                    ) from exc
                raise
            self.mode = f"embedded ({settings.qdrant_path})"
        self._lock = threading.Lock()
        self._is_server = bool(settings.qdrant_url)

    # ------------------------------------------------------------------ admin
    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def exists(self, collection: str) -> bool:
        return self.client.collection_exists(collection)

    def count(self, collection: str) -> int:
        if not self.exists(collection):
            return 0
        return self.client.count(collection, exact=True).count

    def ensure_collection(self, collection: str, dim: int) -> None:
        with self._lock:
            if self.exists(collection):
                info = self.client.get_collection(collection)
                vectors = info.config.params.vectors
                existing = vectors[DENSE].size if isinstance(vectors, dict) else vectors.size
                if existing != dim:
                    raise IndexMismatchError(
                        f"Collection '{collection}' has {existing}-dim vectors, model produces {dim}. "
                        "Delete the collection or use a different collection prefix."
                    )
                return
            log.info("Creating collection %s (dim=%d)", collection, dim)
            self.client.create_collection(
                collection,
                vectors_config={DENSE: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
                sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            )
            if self._is_server:  # speeds up per-document filters; no-op in embedded mode
                self.client.create_payload_index(collection, "doc_id", qm.PayloadSchemaType.KEYWORD)

    def drop_collection(self, collection: str) -> None:
        with self._lock:
            if self.exists(collection):
                self.client.delete_collection(collection)

    # ------------------------------------------------------------------ write
    def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[tuple[list[int], list[float]]],
        batch: int = 128,
    ) -> None:
        points = [
            qm.PointStruct(
                id=c.chunk_id,
                vector={DENSE: d, SPARSE: qm.SparseVector(indices=s[0], values=s[1])},
                payload=c.model_dump(),
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        with self._lock:
            for i in range(0, len(points), batch):
                self.client.upsert(collection, points[i : i + batch], wait=True)

    def delete_document(self, collection: str, doc_id: str) -> None:
        if not self.exists(collection):
            return
        with self._lock:
            self.client.delete(
                collection,
                points_selector=qm.FilterSelector(filter=_doc_filter([doc_id])),
                wait=True,
            )

    # ------------------------------------------------------------------ read
    def hybrid_search(
        self,
        collection: str,
        dense_query: list[float],
        sparse_query: tuple[list[int], list[float]],
        limit: int,
        doc_ids: list[str] | None = None,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        """Run semantic and keyword search, then merge them with Reciprocal Rank
        Fusion (RRF): a chunk ranked high by either method rises to the top.

        ``mode`` = "dense" or "sparse" runs only one of the two (used by the
        evaluation ablation study)."""
        if not self.exists(collection):
            return []
        flt = _doc_filter(doc_ids) if doc_ids else None
        if mode in ("dense", "sparse"):
            query = (dense_query if mode == "dense"
                     else qm.SparseVector(indices=sparse_query[0], values=sparse_query[1]))
            with self._lock:
                res = self.client.query_points(collection, query=query,
                                               using=DENSE if mode == "dense" else SPARSE,
                                               limit=limit, query_filter=flt, with_payload=True)
            return [SearchHit(chunk=Chunk(**p.payload), score=p.score) for p in res.points]
        prefetch = [
            qm.Prefetch(query=dense_query, using=DENSE, limit=limit, filter=flt),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse_query[0], values=sparse_query[1]),
                using=SPARSE, limit=limit, filter=flt,
            ),
        ]
        with self._lock:
            res = self.client.query_points(
                collection,
                prefetch=prefetch,
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        return [SearchHit(chunk=Chunk(**p.payload), score=p.score) for p in res.points]

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def _doc_filter(doc_ids: list[str]) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))])
