"""Ollama chat client with streaming and 'thinking'-model handling."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import ollama

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class ThinkFilter:
    """Removes inline <think>...</think> reasoning from a token stream."""

    def __init__(self) -> None:
        self.inside = False
        self.buffer = ""

    def feed(self, text: str) -> str:
        self.buffer += text
        out = []
        while self.buffer:
            if self.inside:
                end = self.buffer.find("</think>")
                if end == -1:
                    self.buffer = self.buffer[-8:]  # keep a possible partial tag
                    return "".join(out)
                self.buffer = self.buffer[end + len("</think>"):]
                self.inside = False
            else:
                start = self.buffer.find("<think>")
                if start == -1:
                    # Hold back a possible partial "<think" at the end.
                    safe = len(self.buffer) - 7
                    if safe > 0:
                        out.append(self.buffer[:safe])
                        self.buffer = self.buffer[safe:]
                    return "".join(out)
                out.append(self.buffer[:start])
                self.buffer = self.buffer[start + len("<think>"):]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buffer
        self.buffer = ""
        return rest


class OllamaLLM:
    def __init__(self, model: str, url: str, timeout: float, num_ctx: int,
                 temperature: float, think: bool | str, keep_alive: str):
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.think = think
        self.keep_alive = keep_alive
        self.url = url
        self._client = ollama.Client(host=url, timeout=timeout)
        self._think_supported = True

    def _options(self) -> dict:
        return {"num_ctx": self.num_ctx, "temperature": self.temperature}

    def _call(self, messages: list[dict], stream: bool):
        kwargs = dict(model=self.model, messages=messages, stream=stream,
                      options=self._options(), keep_alive=self.keep_alive)
        if self._think_supported:
            kwargs["think"] = self.think
        try:
            return self._client.chat(**kwargs)
        except ollama.ResponseError as exc:
            msg = str(exc).lower()
            if "think" in msg and self._think_supported:
                log.info("Model %s does not support the think option; retrying without it", self.model)
                self._think_supported = False
                return self._call(messages, stream)
            if exc.status_code == 404 or "not found" in msg:
                raise LLMError(
                    f"Model '{self.model}' is not installed on Ollama ({self.url}). "
                    f"Run: ollama pull {self.model}"
                ) from exc
            raise LLMError(f"Ollama error: {exc}") from exc
        except (ConnectionError, OSError) as exc:  # httpx.ConnectError subclasses these
            raise LLMError(
                f"Cannot reach Ollama at {self.url}. Is it installed and running?"
            ) from exc
        except Exception as exc:
            if "connect" in type(exc).__name__.lower() or "connect" in str(exc).lower():
                raise LLMError(
                    f"Cannot reach Ollama at {self.url}. Is it installed and running?"
                ) from exc
            raise

    def chat(self, messages: list[dict]) -> str:
        resp = self._call(messages, stream=False)
        text = resp["message"]["content"] or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def stream(self, messages: list[dict]) -> Iterator[str]:
        filt = ThinkFilter()
        try:
            for part in self._call(messages, stream=True):
                piece = filt.feed(part["message"]["content"] or "")
                if piece:
                    yield piece
        except ollama.ResponseError as exc:
            raise LLMError(f"Ollama error: {exc}") from exc
        tail = filt.flush()
        if tail:
            yield tail
