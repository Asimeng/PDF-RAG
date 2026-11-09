"""A small SQLite catalogue of indexed documents.

It answers "which files are indexed, when, and are they unchanged?" quickly,
without scanning the vector database. The SHA-256 fingerprint of each file
lets us skip re-indexing PDFs that have not changed.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from ragapp.schemas import DocumentInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    collection  TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    file_name   TEXT NOT NULL,
    title       TEXT NOT NULL,
    num_pages   INTEGER NOT NULL,
    num_chunks  INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (collection, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_docs_name ON documents(collection, file_name);
"""


class Registry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _rows(self, sql: str, args: tuple = ()) -> list[DocumentInfo]:
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [DocumentInfo(**dict(r)) for r in rows]

    def list(self, collection: str) -> list[DocumentInfo]:
        return self._rows(
            "SELECT * FROM documents WHERE collection=? ORDER BY file_name", (collection,)
        )

    def get(self, collection: str, doc_id: str) -> DocumentInfo | None:
        rows = self._rows(
            "SELECT * FROM documents WHERE collection=? AND doc_id=?", (collection, doc_id)
        )
        return rows[0] if rows else None

    def find_by_name(self, collection: str, file_name: str) -> list[DocumentInfo]:
        return self._rows(
            "SELECT * FROM documents WHERE collection=? AND file_name=?", (collection, file_name)
        )

    def add(self, collection: str, *, doc_id: str, file_name: str, title: str,
            num_pages: int, num_chunks: int, sha256: str) -> DocumentInfo:
        info = DocumentInfo(
            doc_id=doc_id, file_name=file_name, title=title, num_pages=num_pages,
            num_chunks=num_chunks, sha256=sha256, collection=collection,
            ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
                (collection, info.doc_id, info.file_name, info.title, info.num_pages,
                 info.num_chunks, info.sha256, info.ingested_at),
            )
        return info

    def remove(self, collection: str, doc_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM documents WHERE collection=? AND doc_id=?", (collection, doc_id)
            )

    def collections(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT collection, COUNT(*) AS n FROM documents GROUP BY collection"
            ).fetchall()
        return {r["collection"]: r["n"] for r in rows}

    def close(self) -> None:
        self._conn.close()
