"""A small HTTP client for the REST API, used by the GUI.

The GUI never touches the models or the index directly. It talks to the API like
any other client would, which keeps the GUI thin and lets the API run on a
different machine (e.g. the GPU server) from the one showing the GUI.

Every failure is turned into an ``APIError`` with a message written for
non-technical users, so the GUI can display it as-is.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class APIError(Exception):
    """A problem talking to the API, with a user-friendly message."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RAGClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 900.0,
                 http: httpx.Client | None = None):
        """``http`` lets tests pass a pre-built client (e.g. FastAPI's TestClient)."""
        headers = {"X-API-Key": api_key} if api_key else {}
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(base_url=self.base_url, headers=headers,
                                          timeout=httpx.Timeout(timeout, connect=5.0))

    # ----------------------------------------------------------- plumbing
    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise APIError(
                f"Can't reach the RAG service at {self.base_url}. Is it running? "
                "(Start everything with: rag gui)"
            ) from exc
        except httpx.TimeoutException as exc:
            raise APIError("The service took too long to respond. Please try again.") from exc
        if resp.status_code >= 400:
            raise APIError(_detail(resp), resp.status_code)
        return resp

    def _json(self, method: str, path: str, **kwargs) -> Any:
        return self._request(method, path, **kwargs).json()

    # ----------------------------------------------------------- system
    def health(self) -> bool:
        try:
            return self._http.get("/health", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def status(self) -> dict:
        return self._json("GET", "/status")

    def profiles(self) -> list[dict]:
        return self._json("GET", "/profiles")

    def models(self) -> dict:
        return self._json("GET", "/models")

    def set_llm(self, model: str) -> dict:
        return self._json("PUT", "/config/llm", json={"model": model})

    # ----------------------------------------------------------- documents
    def documents(self) -> list[dict]:
        return self._json("GET", "/documents")

    def upload(self, files: list[tuple[str, bytes]]) -> dict:
        payload = [("files", (name, data, "application/pdf")) for name, data in files]
        return self._json("POST", "/documents", files=payload)

    def job(self, job_id: str) -> dict:
        return self._json("GET", f"/jobs/{job_id}")

    def delete_document(self, doc_id: str, delete_file: bool = False) -> dict:
        return self._json("DELETE", f"/documents/{doc_id}", params={"delete_file": delete_file})

    def document_file(self, doc_id: str) -> bytes:
        return self._request("GET", f"/documents/{doc_id}/file").content

    def page_image(self, doc_id: str, page: int, highlight: str | None = None,
                   zoom: float = 1.6) -> bytes:
        return self._request("POST", f"/documents/{doc_id}/pages/{page}/image",
                             json={"highlight": highlight, "zoom": zoom}).content

    # ----------------------------------------------------------- questions
    def ask_stream(self, question: str, history: list[dict] | None = None,
                   doc_ids: list[str] | None = None) -> Iterator[dict]:
        """Yield the API's streamed events: sources, token..., done (or error)."""
        body = {"question": question, "history": history or None, "doc_ids": doc_ids or None}
        try:
            with self._http.stream("POST", "/ask/stream", json=body) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise APIError(_detail(resp), resp.status_code)
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        yield json.loads(line[6:])
        except httpx.ConnectError as exc:
            raise APIError(f"Can't reach the RAG service at {self.base_url}. Is it running?") from exc
        except httpx.TimeoutException as exc:
            raise APIError("The answer took too long. Try a shorter question or a smaller model.") from exc

    def close(self) -> None:
        self._http.close()


def _detail(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    if resp.status_code == 401:
        return "The service rejected the access key. Check RAG_API_KEY in your settings."
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail or f"The service returned an error ({resp.status_code}).")
