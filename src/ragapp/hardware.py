"""Hardware and Ollama detection used by the ``auto`` profile."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from ragapp.config import PROFILES, Profile, Settings

log = logging.getLogger(__name__)

# Hosts treated as "this machine" (GPU checked with nvidia-smi). Anything else -
# including host.docker.internal seen from inside a container - is "remote".
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


@dataclass
class GPUInfo:
    index: int
    name: str
    total_gb: float
    free_gb: float


def detect_gpus() -> list[GPUInfo]:
    """Return NVIDIA GPUs visible on this machine (empty list if none)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("nvidia-smi failed: %s", exc)
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            gpus.append(GPUInfo(int(parts[0]), parts[1], float(parts[2]) / 1024, float(parts[3]) / 1024))
    return gpus


def normalize_model_name(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def list_ollama_models(url: str, timeout: float = 5.0) -> list[str] | None:
    """Names of models installed on the Ollama server, or None if unreachable."""
    try:
        r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        r.raise_for_status()
        return [normalize_model_name(m["name"]) for m in r.json().get("models", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def missing_models(profile: Profile, installed: list[str]) -> list[str]:
    needed = [profile.llm_model]
    if profile.embed_provider == "ollama":
        needed.append(profile.embed_model)
    have = set(installed)
    return [m for m in needed if normalize_model_name(m) not in have]


def choose_profile(settings: Settings) -> tuple[Profile, str]:
    """Pick the most capable profile this machine + Ollama server can run."""
    gpus = detect_gpus()
    best_free = max((g.free_gb for g in gpus), default=0.0)
    installed = list_ollama_models(settings.ollama_url)
    remote = (urlparse(settings.ollama_url).hostname or "localhost") not in LOCAL_HOSTS

    if installed is None:
        return PROFILES["cpu"], (
            f"auto: Ollama not reachable at {settings.ollama_url}; using 'cpu' profile"
        )

    # Largest first.
    for profile in sorted(PROFILES.values(), key=lambda p: p.min_vram_gb, reverse=True):
        if profile.min_vram_gb == 0:
            continue
        if missing_models(profile, installed):
            continue
        if gpus and best_free >= profile.min_vram_gb:
            return profile, (
                f"auto: GPU with {best_free:.0f} GB free VRAM and required models installed "
                f"-> '{profile.name}'"
            )
        if not gpus and remote:
            # We cannot see the VRAM of a remote server; trust that installed models run.
            return profile, (
                f"auto: remote Ollama ({settings.ollama_url}) has the models for "
                f"'{profile.name}' (VRAM not checked remotely)"
            )

    cpu = PROFILES["cpu"]
    miss = missing_models(cpu, installed)
    note = f" (missing: {', '.join(miss)} - run `ollama pull {miss[0]}`)" if miss else ""
    gpu_note = f"best GPU has {best_free:.0f} GB free" if gpus else "no NVIDIA GPU detected"
    return cpu, f"auto: {gpu_note}; using 'cpu' profile{note}"


def onnx_cuda_available() -> bool:
    """True when onnxruntime can run models on an NVIDIA GPU (onnxruntime-gpu installed)."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


def use_cuda(settings) -> bool:
    """Resolve the ``use_cuda`` setting (None = auto-detect)."""
    if settings.use_cuda is None:
        return onnx_cuda_available()
    return bool(settings.use_cuda)


def is_cloud_model(name: str) -> bool:
    """Ollama cloud models (e.g. ``glm-5.3:cloud``) run remotely on ollama.com."""
    tag = name.lower().rsplit(":", 1)[-1] if ":" in name else ""
    return tag == "cloud" or tag.endswith("-cloud")
