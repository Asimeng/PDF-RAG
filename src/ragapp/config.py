"""Configuration: hardware profiles + user settings.

How configuration works (highest priority first):
  1. Environment variables / `.env` file entries starting with ``RAG_``
     (e.g. ``RAG_LLM_MODEL=qwen3:32b``) override individual values.
  2. The selected *profile* (``RAG_PROFILE=cpu|gpu|gpu-max|auto``) supplies
     sensible defaults for a given class of hardware.

``auto`` inspects the machine (NVIDIA GPU? how much free VRAM?) and the Ollama
server (which models are installed?) and picks the best profile that fits.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

EmbedProvider = Literal["fastembed", "ollama"]


@dataclass(frozen=True)
class Profile:
    """A bundle of model choices tuned for one class of hardware."""

    name: str
    description: str
    llm_model: str
    llm_num_ctx: int
    embed_provider: EmbedProvider
    embed_model: str
    reranker_model: str  # "none" disables reranking
    top_k_retrieve: int  # candidates fetched by hybrid search
    top_k_context: int  # chunks handed to the LLM after reranking
    llm_think: bool | str = False  # reasoning mode for thinking models
    min_vram_gb: float = 0.0  # auto-mode requirement (0 = CPU is fine)


PROFILES: dict[str, Profile] = {
    "cpu": Profile(
        name="cpu",
        description="Laptops / PCs without a GPU. Small, fast models; embeddings run in-process.",
        llm_model="llama3.2:3b",
        llm_num_ctx=4096,
        embed_provider="fastembed",
        embed_model="nomic-ai/nomic-embed-text-v1.5",
        reranker_model="Xenova/ms-marco-MiniLM-L-6-v2",
        top_k_retrieve=20,
        top_k_context=4,
    ),
    "gpu": Profile(
        name="gpu",
        description="A single modern GPU (>= 32 GB free VRAM). Strong MoE model, multilingual embeddings.",
        llm_model="qwen3.6:35b-a3b",
        llm_num_ctx=16384,
        embed_provider="ollama",
        embed_model="qwen3-embedding:0.6b",
        reranker_model="BAAI/bge-reranker-v2-m3",
        top_k_retrieve=40,
        top_k_context=8,
        min_vram_gb=32,
    ),
    "gpu-max": Profile(
        name="gpu-max",
        description="Data-centre GPU (>= 72 GB free VRAM, e.g. H100 80GB). Largest model, best answers.",
        llm_model="gpt-oss:120b",
        llm_num_ctx=32768,
        llm_think="low",
        embed_provider="ollama",
        embed_model="qwen3-embedding:0.6b",
        reranker_model="BAAI/bge-reranker-v2-m3",
        top_k_retrieve=50,
        top_k_context=10,
        min_vram_gb=72,
    ),
}


class Settings(BaseSettings):
    """User-facing settings. Every field can be set via ``RAG_<FIELD>``."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- profile & services -------------------------------------------------
    profile: str = "auto"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout: float = 600.0
    ollama_keep_alive: str = "30m"

    # --- storage ------------------------------------------------------------
    data_dir: Path = Path("data")
    models_dir: Path | None = None  # where downloaded models are cached (default data/models)
    qdrant_url: str | None = None  # e.g. http://qdrant:6333 -> server mode
    qdrant_api_key: str | None = None
    collection_prefix: str = "docs"

    # --- ingestion ----------------------------------------------------------
    chunk_size_tokens: int = 450
    chunk_overlap_tokens: int = 60
    embed_batch_size: int = 32

    # --- optional overrides of the profile (None = use profile value) ------
    llm_model: str | None = None
    llm_num_ctx: int | None = None
    llm_think: bool | str | None = None
    # Ollama ":cloud" models run on ollama.com, so document passages would leave
    # this machine. Blocked unless explicitly allowed.
    allow_cloud_models: bool = False
    llm_temperature: float = 0.1
    embed_provider: EmbedProvider | None = None
    embed_model: str | None = None
    reranker_model: str | None = None
    top_k_retrieve: int | None = None
    top_k_context: int | None = None
    min_rerank_score: float | None = None  # drop weak chunks below this score

    # --- OCR (scanned pages, tables saved as images) -------------------------
    ocr: Literal["auto", "off"] = "auto"
    ocr_dpi: int = 300  # render resolution for OCR; 300 is markedly more accurate than 150

    # --- hardware -----------------------------------------------------------
    # Run fastembed embeddings/reranker on GPU. None = auto (on when onnxruntime-gpu
    # with a working CUDA provider is installed). Needs onnxruntime-gpu.
    use_cuda: bool | None = None
    threads: int | None = None  # CPU threads for fastembed (None = all)

    # --- service ------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: str | None = None  # if set, clients must send header  X-API-Key: <key>
    max_upload_mb: int = 200
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"

    # --- GUI ----------------------------------------------------------------
    gui_host: str = "127.0.0.1"  # 0.0.0.0 to let other computers open the GUI
    gui_port: int = 8501
    api_url: str | None = None  # where the GUI finds the API (default: api_host:api_port)

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def model_cache_dir(self) -> Path:
        return self.models_dir or self.data_dir / "models"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.db"

    @property
    def resolved_api_url(self) -> str:
        if self.api_url:
            return self.api_url.rstrip("/")
        host = "127.0.0.1" if self.api_host in {"0.0.0.0", "::"} else self.api_host
        return f"http://{host}:{self.api_port}"


@dataclass
class RuntimeConfig:
    """The final, fully-resolved configuration the pipeline runs with."""

    profile: str
    profile_reason: str
    llm_model: str
    llm_num_ctx: int
    llm_think: bool | str
    llm_temperature: float
    embed_provider: EmbedProvider
    embed_model: str
    reranker_model: str
    top_k_retrieve: int
    top_k_context: int
    min_rerank_score: float | None
    settings: Settings = field(repr=False)

    @property
    def collection_name(self) -> str:
        """Each embedding model gets its own index (vectors are not interchangeable)."""
        slug = re.sub(r"[^a-z0-9]+", "-", f"{self.embed_provider}-{self.embed_model}".lower())
        return f"{self.settings.collection_prefix}__{slug.strip('-')}"

    def public_dict(self) -> dict:
        d = asdict(self)
        d.pop("settings", None)
        d["collection_name"] = self.collection_name
        return d


def build_runtime_config(settings: Settings, profile: Profile, reason: str) -> RuntimeConfig:
    def pick(override, default):
        return default if override is None else override

    return RuntimeConfig(
        profile=profile.name,
        profile_reason=reason,
        llm_model=pick(settings.llm_model, profile.llm_model),
        llm_num_ctx=pick(settings.llm_num_ctx, profile.llm_num_ctx),
        llm_think=pick(settings.llm_think, profile.llm_think),
        llm_temperature=settings.llm_temperature,
        embed_provider=pick(settings.embed_provider, profile.embed_provider),
        embed_model=pick(settings.embed_model, profile.embed_model),
        reranker_model=pick(settings.reranker_model, profile.reranker_model),
        top_k_retrieve=pick(settings.top_k_retrieve, profile.top_k_retrieve),
        top_k_context=pick(settings.top_k_context, profile.top_k_context),
        min_rerank_score=settings.min_rerank_score,
        settings=settings,
    )


def resolve_config(settings: Settings | None = None) -> RuntimeConfig:
    """Turn settings (+ hardware detection for ``auto``) into a RuntimeConfig."""
    from ragapp.hardware import choose_profile  # local import avoids a cycle

    settings = settings or get_settings()
    requested = settings.profile.lower().strip()
    if requested == "auto":
        profile, reason = choose_profile(settings)
    elif requested in PROFILES:
        profile, reason = PROFILES[requested], f"profile '{requested}' selected explicitly"
    else:
        raise ValueError(
            f"Unknown profile '{settings.profile}'. Choose one of: auto, {', '.join(PROFILES)}"
        )
    return build_runtime_config(settings, profile, reason)


@lru_cache
def get_settings() -> Settings:
    return Settings()
