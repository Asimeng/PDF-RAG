"""Data types shared across the pipeline (also used as API response models)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PageText(BaseModel):
    """Cleaned Markdown text of one PDF page."""

    page: int  # 1-based page number
    text: str


class ParsedDocument(BaseModel):
    doc_id: str
    file_name: str
    title: str
    num_pages: int
    pages: list[PageText]
    ocr_pages: dict[int, str] = {}  # page number -> "page" (scanned) or "table" (table image)
    unread_scanned_pages: int = 0  # scanned pages skipped because no OCR engine is installed


class Chunk(BaseModel):
    """A retrievable piece of a document."""

    chunk_id: str
    doc_id: str
    file_name: str
    title: str
    chunk_index: int
    page_start: int
    page_end: int
    section: str = ""
    text: str  # what the user sees / the LLM reads

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model: prefixed with document context so a
        chunk like 'Results were significant (p<0.01)' still knows what it is about."""
        header = f"Document: {self.title}"
        if self.section:
            header += f"\nSection: {self.section}"
        return f"{header}\n\n{self.text}"

    @property
    def pages_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


class SearchHit(BaseModel):
    chunk: Chunk
    score: float  # fused hybrid-search score
    rerank_score: float | None = None


class Source(BaseModel):
    """A citation shown to the user."""

    ref: int  # the [n] number used in the answer
    doc_id: str = ""
    file_name: str
    title: str
    page_start: int
    page_end: int
    section: str
    text: str
    score: float
    cited: bool = False  # did the answer actually reference this source?


class Answer(BaseModel):
    question: str
    standalone_question: str
    answer: str
    sources: list[Source]
    model: str
    profile: str
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    doc_id: str
    file_name: str
    title: str
    num_pages: int
    num_chunks: int
    sha256: str
    ingested_at: str
    collection: str


class IngestResult(BaseModel):
    file_name: str
    status: str  # "indexed" | "skipped" | "failed"
    doc_id: str | None = None
    num_chunks: int = 0
    message: str = ""
    seconds: float = 0.0
