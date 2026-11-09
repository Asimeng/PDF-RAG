"""Shared test fixtures.

Tests use tiny *fake* embedding and LLM components so they run in seconds,
offline, without Ollama or model downloads. The BM25 encoder is real (small).
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pymupdf
import pytest

from ragapp.config import PROFILES, Settings, build_runtime_config


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors."""

    model_name = "fake"
    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    model = "fake-llm"

    def __init__(self, reply: str = "Photosynthesis converts light into chemical energy [1]."):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages):
        self.calls.append(messages)
        return self.reply

    def stream(self, messages):
        self.calls.append(messages)
        for word in self.reply.split(" "):
            yield word + " "


def make_pdf(path: Path, pages: list[str], header: str | None = None) -> Path:
    doc = pymupdf.open()
    for i, text in enumerate(pages, 1):
        page = doc.new_page()
        if header:  # running header at the top, page number at the bottom (like journals)
            page.insert_text((50, 30), header, fontsize=8)
            page.insert_text((300, 820), str(i), fontsize=8)
        page.insert_textbox(pymupdf.Rect(50, 60, 550, 800), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data", profile="cpu",
                    models_dir=Path(__file__).resolve().parents[1] / "data" / "models",
                    chunk_size_tokens=120, chunk_overlap_tokens=20)


@pytest.fixture
def runtime_cfg(settings):
    return build_runtime_config(settings, PROFILES["cpu"], "test")


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pages = [
        "Plant Biology Basics\n\nPhotosynthesis is the process by which plants convert "
        "light energy into chemical energy stored in glucose. It takes place in the chloroplasts.\n\n"
        "Chlorophyll is the green pigment that absorbs mostly blue and red light.",
        "Cellular respiration breaks down glucose to release ATP in the mitochondria. "
        "Oxygen is consumed and carbon dioxide is released.\n\nThe Krebs cycle is a key stage.",
        "The enzyme RuBisCO fixes carbon dioxide during the Calvin cycle. RuBisCO is "
        "considered the most abundant protein on Earth.",
        "References\n\nSmith J. Photosynthesis photosynthesis photosynthesis. 2001.",
    ]
    return make_pdf(tmp_path / "biology.pdf", pages, header="Journal of Tests 2024")


@pytest.fixture
def pipeline(runtime_cfg):
    from ragapp.pipeline import RAGPipeline

    pipe = RAGPipeline(runtime_cfg, embedder=FakeEmbedder(), llm=FakeLLM(), reranker=False)
    yield pipe
    pipe.close()
