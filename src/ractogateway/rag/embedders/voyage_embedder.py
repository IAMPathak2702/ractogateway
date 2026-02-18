"""Voyage AI embedding provider (Anthropic-aligned, best for Claude RAG).

Install with:  pip install ractogateway[rag-voyage]
"""

from __future__ import annotations

import os
from typing import Any


def _require_voyage() -> Any:
    try:
        import voyageai  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "VoyageEmbedder requires the 'voyageai' package. "
            "Install it with:  pip install ractogateway[rag-voyage]"
        ) from exc
    return voyageai


from ractogateway.rag.embedders.base import BaseEmbedder

_KNOWN_DIMS: dict[str, int] = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-3-large": 2048,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
    "voyage-multilingual-2": 1024,
}


class VoyageEmbedder(BaseEmbedder):
    """Embed texts using the Voyage AI API.

    Voyage AI embeddings are optimised for Anthropic Claude RAG pipelines
    and are the recommended choice when using Claude as the generation LLM.

    Parameters
    ----------
    model:
        Voyage model name (default ``"voyage-3"``).
    api_key:
        Voyage API key.  Falls back to ``VOYAGE_API_KEY`` env var.
    input_type:
        ``"query"`` for queries, ``"document"`` for documents to index.
        Using the correct type improves retrieval quality.
    batch_size:
        Maximum texts per API call.
    """

    def __init__(
        self,
        model: str = "voyage-3",
        *,
        api_key: str | None = None,
        input_type: str | None = "document",
        batch_size: int = 128,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self._input_type = input_type
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._model, -1)

    def _make_client(self) -> Any:
        voyageai = _require_voyage()
        if self._api_key:
            return voyageai.Client(api_key=self._api_key)
        return voyageai.Client()

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._make_client()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            kw: dict[str, Any] = {"model": self._model, "input": batch}
            if self._input_type:
                kw["input_type"] = self._input_type
            response = client.embed(**kw)
            results.extend(response.embeddings)
        return results

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        voyageai = _require_voyage()
        if self._api_key:
            client = voyageai.AsyncClient(api_key=self._api_key)
        else:
            client = voyageai.AsyncClient()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            kw: dict[str, Any] = {"model": self._model, "input": batch}
            if self._input_type:
                kw["input_type"] = self._input_type
            response = await client.embed(**kw)
            results.extend(response.embeddings)
        return results
