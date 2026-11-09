"""Evaluation harness - measure how well the system works, with numbers.

Retrieval metrics (no LLM needed, fast):
  * Hit@k  - share of questions where a correct passage (right file + page) is
             among the top k results.
  * MRR    - Mean Reciprocal Rank: 1.0 if the correct passage is ranked first,
             0.5 if second, 0.33 if third ... averaged over questions.
  Computed for 4 configurations (an *ablation study*) to show what each
  component contributes: dense-only, BM25-only, hybrid, hybrid + reranker.

Answer metrics (needs Ollama, slower):
  * Key-fact recall    - answer contains the expected key facts (tolerant of
                         formatting: "7" = "seven", "STAT 3" = "STAT3").
  * Citation rate      - answer cites at least one source.
  * Refusal accuracy   - for questions NOT answerable from the documents, the
                         system correctly says it could not find the answer.
  * Faithfulness       - (optional, --judge MODEL) an LLM judge checks that the
                         answer is fully supported by the retrieved sources.
"""

from __future__ import annotations

import json
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ragapp.generate.llm import OllamaLLM
from ragapp.pipeline import RAGPipeline
from ragapp.schemas import SearchHit

MODES = {
    "dense": dict(mode="dense", rerank=False),
    "bm25": dict(mode="sparse", rerank=False),
    "hybrid": dict(mode="hybrid", rerank=False),
    "hybrid+rerank": dict(mode="hybrid", rerank=True),
}

# The judge must reason claim by claim before deciding. A one-word verdict with
# no reasoning proved far too strict in practice: it rejected correct answers
# that merely reworded the source ("Illumina NovaSeq S2" vs "NovaSeq S2 (Illumina)").
JUDGE_PROMPT = """You are checking whether an ANSWER is supported by SOURCES (passages from documents).

Check each factual claim in the ANSWER against the SOURCES:
- Supported: the sources state it or directly imply it. Paraphrases, reordering, rounding, number formats ("7" vs "seven", "STAT 3" vs "STAT3") and details such as a vendor name that appear in the sources all count as supported.
- Unsupported: the sources do not contain the claim, or contradict it.
Ignore citation markers such as [1].

SOURCES:
{sources}

QUESTION: {question}

ANSWER: {answer}

Briefly list any unsupported claims (or write "none"). Then finish with exactly one line:
VERDICT: SUPPORTED   or   VERDICT: UNSUPPORTED"""

_VERDICT = re.compile(r"VERDICT\W*(UNSUPPORTED|SUPPORTED)", re.I)


def parse_verdict(text: str) -> bool | None:
    """True/False from the judge's reply; None if it gave no recognisable verdict."""
    found = _VERDICT.findall(text or "")
    if found:
        return found[-1].upper() == "SUPPORTED"
    words = re.findall(r"\b(UNSUPPORTED|SUPPORTED)\b", (text or "").upper())
    return (words[-1] == "SUPPORTED") if words else None


_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[‐-―−]", "-", text)  # unicode hyphens/dashes
    return _NUMBER_WORD_RE.sub(lambda m: _NUMBER_WORDS[m.group(1)], text)


def fact_present(fact: str, answer: str) -> bool:
    """Is a key fact in the answer, ignoring case, spacing, hyphens and number words?"""
    f, a = _normalize(fact), _normalize(answer)
    if re.fullmatch(r"[\d.,]+", f):  # a number: match it as a whole number
        f_digits = re.sub(r"[.,](?=\d{3}\b)", "", f)
        a_digits = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", a)
        return re.search(rf"(?<![\d.]){re.escape(f_digits)}(?![\d])", a_digits) is not None
    squash = lambda t: re.sub(r"[\W_]+", "", t)  # noqa: E731
    return squash(f) in squash(a)


@dataclass
class EvalItem:
    id: str
    question: str
    file: str | None = None
    pages: list[int] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    expect_not_found: bool = False


def load_questions(path: Path) -> list[EvalItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("//"):
            items.append(EvalItem(**json.loads(line)))
    return items


def _is_relevant(hit: SearchHit, item: EvalItem) -> bool:
    c = hit.chunk
    return c.file_name == item.file and any(c.page_start <= p <= c.page_end for p in item.pages)


def evaluate_retrieval(pipe: RAGPipeline, items: list[EvalItem], k: int) -> dict:
    answerable = [i for i in items if not i.expect_not_found and i.file]
    results = {}
    for name, kwargs in MODES.items():
        if kwargs["rerank"] and pipe.reranker is None:
            continue
        ranks, t0 = [], time.perf_counter()
        for item in answerable:
            hits = pipe.retriever.retrieve(item.question, top_k=k, **kwargs)
            rank = next((n for n, h in enumerate(hits, 1) if _is_relevant(h, item)), None)
            ranks.append(rank)
        n = len(answerable) or 1
        results[name] = {
            "hit@1": sum(1 for r in ranks if r == 1) / n,
            "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
            f"hit@{k}": sum(1 for r in ranks if r) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
            "ms_per_query": (time.perf_counter() - t0) * 1000 / n,
            "misses": [i.id for i, r in zip(answerable, ranks) if not r],
        }
    return results


def _refused(answer: str) -> bool:
    a = answer.lower()
    return "couldn't find" in a or "could not find" in a or "not contain" in a


def evaluate_answers(pipe: RAGPipeline, items: list[EvalItem], judge_model: str | None) -> dict:
    judge = None
    if judge_model:
        s = pipe.cfg.settings
        judge = OllamaLLM(judge_model, s.ollama_url, s.ollama_timeout, 16384, 0.0, None, s.ollama_keep_alive)
    rows = []
    for item in items:
        t0 = time.perf_counter()
        ans = pipe.ask(item.question)
        secs = time.perf_counter() - t0
        row = {"id": item.id, "seconds": round(secs, 1), "answer": ans.answer,
               "cited": any(s.cited for s in ans.sources), "refused": _refused(ans.answer)}
        if item.expect_not_found:
            row["correct_refusal"] = row["refused"]
        else:
            row["facts_found"] = all(fact_present(f, ans.answer) for f in item.must_include)
        if judge and not item.expect_not_found and ans.sources:
            src = "\n\n".join(f"[{s.ref}] {s.text}" for s in ans.sources)
            verdict = judge.chat([{"role": "user", "content": JUDGE_PROMPT.format(
                sources=src, question=item.question, answer=ans.answer)}])
            verdict_ok = parse_verdict(verdict)
            if verdict_ok is not None:
                row["faithful"] = verdict_ok
            row["judge_note"] = verdict.strip()[-400:]  # kept in the JSON report for auditing
        rows.append(row)

    pos = [r for r in rows if "facts_found" in r]
    neg = [r for r in rows if "correct_refusal" in r]
    judged = [r for r in pos if "faithful" in r]

    def rate(xs, key):
        return sum(1 for x in xs if x[key]) / len(xs) if xs else None

    return {
        "key_fact_recall": rate(pos, "facts_found"),
        "citation_rate": rate(pos, "cited"),
        "false_refusal_rate": rate(pos, "refused"),
        "refusal_accuracy": rate(neg, "correct_refusal"),
        "faithfulness": rate(judged, "faithful"),
        "median_seconds": round(statistics.median(r["seconds"] for r in rows), 1) if rows else None,
        "rows": rows,
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def render_report(cfg: dict, retrieval: dict, answers: dict | None, k: int, n: int) -> str:
    lines = [
        f"# Evaluation report - {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- Profile: **{cfg['profile']}** | LLM: `{cfg['llm_model']}` | "
        f"Embeddings: `{cfg['embed_provider']}:{cfg['embed_model']}` | Reranker: `{cfg['reranker_model']}`",
        f"- Questions: {n}",
        "",
        "## Retrieval (ablation study)",
        "",
        f"| Configuration | Hit@1 | Hit@3 | Hit@{k} | MRR | ms/query |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in retrieval.items():
        lines.append(f"| {name} | {_pct(r['hit@1'])} | {_pct(r['hit@3'])} | {_pct(r[f'hit@{k}'])} "
                     f"| {r['mrr']:.3f} | {r['ms_per_query']:.0f} |")
    misses = retrieval.get("hybrid+rerank", next(iter(retrieval.values()), {})).get("misses", [])
    if misses:
        lines += ["", f"Missed by the final configuration: {', '.join(misses)}"]
    if answers:
        lines += [
            "", "## Answers", "",
            "| Metric | Value |", "|---|---|",
            f"| Key-fact recall | {_pct(answers['key_fact_recall'])} |",
            f"| Citation rate | {_pct(answers['citation_rate'])} |",
            f"| Refusal accuracy (unanswerable questions) | {_pct(answers['refusal_accuracy'])} |",
            f"| False refusals (answerable questions) | {_pct(answers['false_refusal_rate'])} |",
            f"| Faithfulness (LLM judge) | {_pct(answers['faithfulness'])} |",
            f"| Median seconds per answer | {answers['median_seconds']} |",
            "", "<details><summary>Individual answers</summary>", "",
        ]
        for r in answers["rows"]:
            flags = ", ".join(f"{k}={v}" for k, v in r.items() if k not in {"id", "answer", "seconds", "judge_note"})
            lines += [f"**{r['id']}** ({r['seconds']}s; {flags})", "", "> " + r["answer"].replace("\n", "\n> "), ""]
        lines.append("</details>")
    return "\n".join(lines) + "\n"


def run(questions: Path, out_dir: Path, generate: bool, judge_model: str | None) -> Path:
    pipe = RAGPipeline()
    try:
        items = load_questions(questions)
        k = pipe.cfg.top_k_context
        retrieval = evaluate_retrieval(pipe, items, k)
        answers = evaluate_answers(pipe, items, judge_model) if generate else None
        cfg = pipe.cfg.public_dict()
    finally:
        pipe.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M}_{cfg['profile']}"
    report = out_dir / f"{stamp}.md"
    report.write_text(render_report(cfg, retrieval, answers, k, len(items)), encoding="utf-8")
    (out_dir / f"{stamp}.json").write_text(
        json.dumps({"config": cfg, "retrieval": retrieval, "answers": answers}, indent=2), encoding="utf-8")
    return report
