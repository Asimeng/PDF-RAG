from ragapp.generate.llm import ThinkFilter
from ragapp.generate.prompts import cited_refs, format_sources, normalize_citations
from ragapp.schemas import Chunk, SearchHit


def test_cited_refs_formats():
    assert cited_refs("A [1]. B [2, 3]. C [4-6]. D [1][7]") == {1, 2, 3, 4, 5, 6, 7}
    assert cited_refs("no citations here") == set()


def _hit(text: str, i: int = 0) -> SearchHit:
    return SearchHit(score=1.0, chunk=Chunk(chunk_id=str(i), doc_id="d", file_name="f.pdf",
                                            title="T", chunk_index=i, page_start=1, page_end=2,
                                            section="S", text=text))


def test_format_sources_respects_budget():
    hits = [_hit("word " * 200, i) for i in range(10)]
    text, used = format_sources(hits, token_budget=600)
    assert 1 <= len(used) < 10
    assert text.startswith("[1] f.pdf, pp. 1-2 | Section: S")


def test_think_filter_streaming():
    f = ThinkFilter()
    pieces = ["Hel", "lo <thi", "nk>secret reasoning</th", "ink> world", "!"]
    out = "".join(f.feed(p) for p in pieces) + f.flush()
    assert out == "Hello  world!"


def test_normalize_citations_variants():
    # gpt-oss style fullwidth brackets with line markers, and "[Source n]".
    text = "A 【1†L3-L7】. B [Source 2]. C 【3，4】. D [5†source]."
    assert normalize_citations(text) == "A [1]. B [2]. C [3, 4]. D [5]."
    assert cited_refs(normalize_citations(text)) == {1, 2, 3, 4, 5}
    assert normalize_citations("plain [1] stays") == "plain [1] stays"
