import pytest

from ragapp.evaluate import fact_present, parse_verdict


@pytest.mark.parametrize("fact, answer", [
    ("seven", "The study profiled 7 hospitalized patients"),
    ("STAT3", "activation shifts from STAT 1 to STAT 3"),
    ("STAT3", "a STAT‑3‑driven response"),           # non-breaking hyphen (gpt-oss)
    ("48,266", "48,266 single‑cell transcriptomes"),
    ("48,266", "48266 transcriptomes"),
    ("NovaSeq", "an Illumina novaseq S2 instrument"),
    ("developing neutrophils", "a population of **developing neutrophils**"),
])
def test_fact_present_tolerates_formatting(fact, answer):
    assert fact_present(fact, answer)


@pytest.mark.parametrize("fact, answer", [
    ("six", "16 donors"),               # whole numbers only
    ("48,266", "148,266 cells"),
    ("STAT3", "STAT1 only"),
])
def test_fact_present_rejects_near_misses(fact, answer):
    assert not fact_present(fact, answer)


def test_parse_verdict():
    assert parse_verdict("Unsupported claims: none\nVERDICT: SUPPORTED") is True
    assert parse_verdict("- 'S2' not in sources\nVERDICT: UNSUPPORTED") is False
    assert parse_verdict("**VERDICT:** SUPPORTED") is True
    assert parse_verdict("SUPPORTED") is True
    assert parse_verdict("I am not sure.") is None
