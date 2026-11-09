import pytest

from ragapp import hardware
from ragapp.config import PROFILES, Settings, resolve_config


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_explicit_profile_and_overrides():
    cfg = resolve_config(_settings(profile="gpu", llm_model="qwen3:32b", top_k_context=3))
    assert cfg.profile == "gpu"
    assert cfg.llm_model == "qwen3:32b"  # override wins
    assert cfg.top_k_context == 3
    assert cfg.embed_model == PROFILES["gpu"].embed_model  # profile default kept


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        resolve_config(_settings(profile="quantum"))


def test_collection_name_depends_on_embedding_model():
    a = resolve_config(_settings(profile="cpu"))
    b = resolve_config(_settings(profile="gpu"))
    assert a.collection_name != b.collection_name
    assert a.collection_name.startswith("docs__")


def _fake_env(monkeypatch, gpus, models):
    monkeypatch.setattr(hardware, "detect_gpus", lambda: gpus)
    monkeypatch.setattr(hardware, "list_ollama_models", lambda url, timeout=5.0: models)


def test_auto_picks_gpu_max_on_big_gpu(monkeypatch):
    _fake_env(monkeypatch, [hardware.GPUInfo(0, "H100", 80, 79)],
              ["gpt-oss:120b", "qwen3-embedding:0.6b", "qwen3.6:35b-a3b"])
    assert resolve_config(_settings(profile="auto")).profile == "gpu-max"


def test_auto_falls_back_when_vram_busy(monkeypatch):
    _fake_env(monkeypatch, [hardware.GPUInfo(0, "H100", 80, 40)],
              ["gpt-oss:120b", "qwen3-embedding:0.6b", "qwen3.6:35b-a3b"])
    assert resolve_config(_settings(profile="auto")).profile == "gpu"


def test_auto_falls_back_when_models_missing(monkeypatch):
    _fake_env(monkeypatch, [hardware.GPUInfo(0, "H100", 80, 79)], ["llama3.2:3b"])
    assert resolve_config(_settings(profile="auto")).profile == "cpu"


def test_auto_cpu_without_ollama(monkeypatch):
    _fake_env(monkeypatch, [], None)
    cfg = resolve_config(_settings(profile="auto"))
    assert cfg.profile == "cpu" and "not reachable" in cfg.profile_reason


def test_auto_trusts_remote_ollama(monkeypatch):
    _fake_env(monkeypatch, [], ["qwen3.6:35b-a3b", "qwen3-embedding:0.6b"])
    cfg = resolve_config(_settings(profile="auto", ollama_url="http://gpu-box:11434"))
    assert cfg.profile == "gpu"


def test_cloud_models_detected():
    from ragapp.hardware import is_cloud_model

    assert is_cloud_model("glm-5.3:cloud") and is_cloud_model("gpt-oss:120b-cloud")
    assert not is_cloud_model("gpt-oss:120b") and not is_cloud_model("llama3.2")
