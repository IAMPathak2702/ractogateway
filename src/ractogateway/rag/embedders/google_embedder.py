"""Google Gemini embedding provider.

Install with:  pip install ractogateway[google]
"""

from __future__ import annotations

import os
from typing import Any


def _require_google() -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "GoogleEmbedder requires the 'google-genai' package. "
            "Install it with:  pip install ractogateway[google]"
        ) from exc
    return genai


from ractogateway.rag.embedders.base import BaseEmbedder

_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-004": 768,
    "embedding-001": 768,
}


class GoogleEmbedder(BaseEmbedder):
    """Embed texts using the Google Gemini Embeddings API.

    Parameters
    ----------
    model:
        Gemini embedding model (default ``"text-embedding-004"``).
    api_key:
        Gemini API key.  Falls back to ``GEMINI_API_KEY`` env var.
    task_type:
        Gemini task type hint (e.g. ``"RETRIEVAL_DOCUMENT"``,
        ``"RETRIEVAL_QUERY"``).  ``None`` lets the API decide.
    batch_size:
        Maximum number of texts per API call.
    """

    def __init__(
        self,
        model: str = "text-embedding-004",
        *,
        api_key: str | None = None,
        task_type: str | None = None,
        batch_size: int = 100,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._task_type = task_type
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._model, -1)

    def _make_client(self) -> Any:
        genai = _require_google()
        kw: dict[str, Any] = {}
        if self._api_key:
            kw["api_key"] = self._api_key
        return genai.Client(**kw)

    def _embed_batch(self, client: Any, batch: list[str]) -> list[list[float]]:
        kw: dict[str, Any] = {"model": self._model, "contents": batch}
        if self._task_type:
            kw["config"] = {"task_type": self._task_type}
        response = client.models.embed_content(**kw)
        return [emb.values for emb in response.embeddings]

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._make_client()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._embed_batch(client, batch))
        return results

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        # google-genai async client mirrors the sync API
        genai = _require_google()
        kw: dict[str, Any] = {}
        if self._api_key:
            kw["api_key"] = self._api_key
        client = genai.AsyncClient(**kw)
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embed_kw: dict[str, Any] = {"model": self._model, "contents": batch}
            if self._task_type:
                embed_kw["config"] = {"task_type": self._task_type}
            response = await client.models.embed_content(**embed_kw)
            results.extend(emb.values for emb in response.embeddings)
        return results
