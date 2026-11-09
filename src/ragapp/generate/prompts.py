"""Prompt templates and citation helpers.

The prompt is what turns a general chatbot into a *grounded* assistant: it must
answer only from the supplied sources, cite them, and admit when it can't.
"""

from __future__ import annotations

import re

from ragapp.ingest.chunker import estimate_tokens
from ragapp.schemas import SearchHit

NOT_FOUND = "I couldn't find the answer to that in the provided documents."

SYSTEM_PROMPT = f"""You are a careful assistant that answers questions using ONLY the numbered sources provided.

Rules:
1. Base every statement on the sources. Do not use outside knowledge.
2. Cite the sources you use with their number in plain square brackets, e.g. [1] or [2][3], right after the statement they support. Use only this format: no other bracket styles, line numbers or source names.
3. If the sources do not contain the answer, reply exactly: "{NOT_FOUND}" You may then briefly mention what related information the sources do contain.
4. Be clear and concise. Use bullet points for lists. Explain technical terms simply when helpful.
5. Answer in the same language as the question."""

CONDENSE_PROMPT = """Given the conversation and a follow-up question, rewrite the follow-up into a single standalone question that can be understood without the conversation. Keep names, numbers and technical terms. Reply with the rewritten question only.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""


def format_sources(hits: list[SearchHit], token_budget: int) -> tuple[str, list[SearchHit]]:
    """Render hits as numbered sources, stopping before the context budget is exceeded."""
    parts, used, total = [], [], 0
    for hit in hits:
        c = hit.chunk
        header = f"[{len(used) + 1}] {c.file_name}, {c.pages_label}"
        if c.section:
            header += f" | Section: {c.section}"
        block = f"{header}\n{c.text}"
        t = estimate_tokens(block)
        if used and total + t > token_budget:
            break
        parts.append(block)
        used.append(hit)
        total += t
    return "\n\n---\n\n".join(parts), used


def build_messages(question: str, sources_text: str) -> list[dict]:
    user = f"Sources:\n\n{sources_text}\n\n=====\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def format_history(history: list[dict], max_turns: int = 6) -> str:
    lines = []
    for turn in history[-max_turns:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = re.sub(r"\s+", " ", str(turn.get("content", "")))[:800]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# Alternative citation styles some models produce, e.g. gpt-oss writes
# fullwidth brackets with a dagger ("【1†L3-L7】"); others write
# "[Source 1]" or "[1†source]". Escapes keep this file ASCII-only.
_ALT_CITE = re.compile(
    r"[\[【]\s*(?:sources?|refs?|doc(?:ument)?s?)?\s*#?\s*"
    r"(\d+(?:\s*[,;，\-–]\s*(?:sources?\s*)?\d+)*)"
    r"(?:\s*†[^\]】\n]*)?\s*[\]】]",
    re.IGNORECASE,
)


def normalize_citations(answer: str) -> str:
    """Rewrite citation variants to the canonical ``[n]`` form."""

    def repl(m: re.Match) -> str:
        nums = re.sub(r"(?i)sources?\s*", "", m.group(1))
        nums = re.sub(r"\s*[,;，]\s*", ", ", nums)
        return "[" + nums.strip() + "]"

    return _ALT_CITE.sub(repl, answer)


_CITE = re.compile(r"\[(\d+(?:\s*[,;\-–]\s*\d+)*)\]")


def cited_refs(answer: str) -> set[int]:
    """Extract citation numbers: [1], [2, 3], [4-6]."""
    refs: set[int] = set()
    for m in _CITE.finditer(answer):
        for part in re.split(r"\s*[,;]\s*", m.group(1)):
            if re.search(r"[\-–]", part):
                a, b = re.split(r"\s*[\-–]\s*", part)[:2]
                if a.isdigit() and b.isdigit() and int(b) - int(a) < 50:
                    refs.update(range(int(a), int(b) + 1))
            elif part.isdigit():
                refs.add(int(part))
    return refs
