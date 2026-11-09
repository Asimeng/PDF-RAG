"""Structure-aware chunking.

Why chunk at all? An LLM can only read a limited amount of text, and search is
more precise over small, focused passages than over whole documents.

Strategy:
  * Split each page into *blocks* (paragraphs, headings, tables, list groups).
  * Headings start a new *section*; the section name is stored with each chunk.
  * Blocks are packed into chunks of ~``chunk_size`` tokens; consecutive chunks
    share ~``overlap`` tokens so a sentence cut at a boundary is not lost.
  * Oversized paragraphs are split on sentence boundaries; oversized tables are
    split on row boundaries with the header row repeated.
  * Reference lists / bibliographies are skipped (they match almost any query
    lexically but rarely answer anything).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from ragapp.schemas import Chunk, ParsedDocument

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_BOLD_HEADING = re.compile(r"^\*\*([^*]{3,120})\*\*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_SKIP_SECTIONS = re.compile(
    r"^(references?|bibliography|works cited|literature cited|reference list)$", re.I
)

# Short "statement" sections that cover only the paragraph right after them.
# Without this, a "Data availability" note on page 1 of a paper whose main text
# has no headings would label every later passage "Data availability".
_STATEMENT_SECTIONS = re.compile(
    r"^(abstract|keywords?|data (and code )?availability|code availability|"
    r"(data|code) availability statement|competing interests?|conflicts? of interests?|"
    r"declaration of interests?|author contributions?|acknowledge?ments?|funding|"
    r"ethics( statement)?|abbreviations)$", re.I
)

# A bibliography entry: optional number, then "Surname, A.B.," ... "(2020)." or "et al."
_REF_START = re.compile(r"^(?:\[?\d{1,3}[.\])]\s+)?[A-Z][\w'´`¨\-]+(?:\s[\w'´`¨\-]+)?,\s(?:[A-Z]\.\s?-?)+[,;]?")
_REF_YEAR = re.compile(r"\((?:19|20)\d{2}[a-z]?\)\.|\bet al\.")

NAMESPACE = uuid.UUID("6f1c1e5e-9a43-4c47-9d0b-8a1f5d7f0c11")


def looks_like_reference(para: str) -> bool:
    """True if most lines/entries of a paragraph look like bibliography entries.
    Catches reference lists even when the PDF lacks a 'References' heading."""
    entries = [ln.strip() for ln in re.split(r"\n|(?<=\.)\s(?=(?:\d{1,3}\.\s)?[A-Z][\w'´`¨\-]+,\s[A-Z]\.)", para)
               if ln.strip()]
    if not entries:
        return False
    hits = sum(1 for e in entries if _REF_START.match(e) and _REF_YEAR.search(e))
    starts = sum(1 for e in entries if _REF_START.match(e))
    return hits >= 1 and (hits + starts) / (2 * len(entries)) >= 0.5


def estimate_tokens(text: str) -> int:
    """Fast approximation: ~4 characters per token for English text."""
    return max(1, len(text) // 4)


@dataclass
class Block:
    text: str
    page: int
    section: str
    is_heading: bool = False


def _heading_text(line: str) -> str | None:
    m = _HEADING.match(line) or _BOLD_HEADING.match(line.strip())
    if not m:
        return None
    text = m.group(m.lastindex).strip().strip("*").strip()
    # Ignore logo fragments such as the journal mark "ll" rendered as a heading.
    return text if len(re.sub(r"\W", "", text)) >= 3 else None


def split_blocks(doc: ParsedDocument, skip_references: bool = True) -> list[Block]:
    blocks: list[Block] = []
    section = ""
    skipping = False
    scoped = False  # current section is a one-paragraph statement section
    for page in doc.pages:
        for para in re.split(r"\n\s*\n", page.text):
            para = para.strip()
            if not para:
                continue
            first_line = para.splitlines()[0]
            if _HEADING.match(first_line) and _heading_text(first_line) is None:
                # A junk heading line (e.g. "### **ll**"): drop that line, keep the rest.
                para = "\n".join(para.splitlines()[1:]).strip()
                if not para:
                    continue
                first_line = para.splitlines()[0]
            heading = _heading_text(first_line) if len(first_line) < 150 else None
            if heading:
                section = heading
                scoped = bool(_STATEMENT_SECTIONS.match(heading))
                skipping = skip_references and bool(_SKIP_SECTIONS.match(heading))
                if skipping:
                    continue
                blocks.append(Block(first_line.strip(), page.page, section, is_heading=True))
                rest = "\n".join(para.splitlines()[1:]).strip()
                if rest:
                    blocks.append(Block(rest, page.page, section))
                    if scoped:  # the statement was this paragraph; the section ends here
                        section, scoped = "", False
                continue
            if not skipping and not (skip_references and looks_like_reference(para)):
                blocks.append(Block(para, page.page, section))
                if scoped:
                    section, scoped = "", False
    return blocks


def _split_oversized(block: Block, max_tokens: int) -> list[Block]:
    if estimate_tokens(block.text) <= max_tokens:
        return [block]
    lines = block.text.splitlines()
    if len(lines) > 2 and all(ln.lstrip().startswith("|") for ln in lines[:2]):
        # Markdown table: split by rows, repeat header (+ separator row).
        header = lines[:2]
        pieces, current = [], list(header)
        for row in lines[2:]:
            if estimate_tokens("\n".join(current + [row])) > max_tokens and len(current) > 2:
                pieces.append("\n".join(current))
                current = list(header)
            current.append(row)
        pieces.append("\n".join(current))
    else:
        sentences = _SENTENCE.split(block.text)
        pieces, current = [], ""
        for s in sentences:
            if current and estimate_tokens(current + " " + s) > max_tokens:
                pieces.append(current)
                current = s
            else:
                current = f"{current} {s}".strip()
        if current:
            pieces.append(current)
        # A single giant "sentence" (no punctuation) - hard split by characters.
        hard = []
        for p in pieces:
            step = max_tokens * 4
            hard.extend(p[i : i + step] for i in range(0, len(p), step))
        pieces = hard
    return [Block(p, block.page, block.section) for p in pieces]


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = 450,
    overlap: int = 60,
    skip_references: bool = True,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    blocks: list[Block] = []
    for b in split_blocks(doc, skip_references):
        blocks.extend(_split_oversized(b, chunk_size))

    groups: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for b in blocks:
        t = estimate_tokens(b.text)
        # Start a fresh chunk at a heading if the current one is already substantial.
        new_section = b.is_heading and size > chunk_size * 0.3
        if current and (size + t > chunk_size or new_section):
            groups.append(current)
            # Carry trailing blocks forward as overlap (never a lone heading).
            carry, carry_size = [], 0
            if not new_section:
                for prev in reversed(current):
                    pt = estimate_tokens(prev.text)
                    if carry_size + pt > overlap:
                        break
                    carry.insert(0, prev)
                    carry_size += pt
            current, size = carry, carry_size
        current.append(b)
        size += t
    if current:
        groups.append(current)

    # Merge tiny groups (e.g. a short section) into their predecessor.
    min_size = chunk_size * 0.15
    merged: list[list[Block]] = []
    for group in groups:
        g_size = sum(estimate_tokens(b.text) for b in group)
        prev_size = sum(estimate_tokens(b.text) for b in merged[-1]) if merged else 0
        if merged and g_size < min_size and prev_size + g_size <= chunk_size * 1.25:
            merged[-1].extend(blk for blk in group if blk not in merged[-1])
        else:
            merged.append(group)
    groups = merged

    chunks: list[Chunk] = []
    for i, group in enumerate(groups):
        text = "\n\n".join(b.text for b in group).strip()
        if not text or all(b.is_heading for b in group):
            continue
        section = next((b.section for b in reversed(group) if b.section), "")
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid5(NAMESPACE, f"{doc.doc_id}:{len(chunks)}")),
                doc_id=doc.doc_id,
                file_name=doc.file_name,
                title=doc.title,
                chunk_index=len(chunks),
                page_start=min(b.page for b in group),
                page_end=max(b.page for b in group),
                section=section,
                text=text,
            )
        )
    return chunks
