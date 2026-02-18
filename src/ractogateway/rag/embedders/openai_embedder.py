"""OpenAI embedding provider.

Install with:  pip install ractogateway[openai]
"""

from __future__ import annotations

import os
from typing import Any


def _require_openai() -> Any:
    try:
        import openai  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "OpenAIEmbedder requires the 'openai' package. "
            "Install it with:  pip install ractogateway[openai]"
        ) from exc
    return openai


from ractogateway.rag.embedders.base import BaseEmbedder

# Dimension lookup for well-known models
_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """Embed texts using the OpenAI Embeddings API.

    Parameters
    ----------
    model:
        OpenAI embedding model (default ``"text-embedding-3-small"``).
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    base_url:
        Custom base URL (Azure OpenAI or proxy).
    dimensions:
        Override output dimensionality (supported for ``text-embedding-3-*``).
    batch_size:
        Maximum number of texts per API call.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 256,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url
        self._dimensions = dimensions
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        if self._dimensions is not None:
            return self._dimensions
        return _KNOWN_DIMS.get(self._model, -1)

    def _client_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if self._api_key:
            kw["api_key"] = self._api_key
        if self._base_url:
            kw["base_url"] = self._base_url
        return kw

    def _call_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if self._dimensions is not None:
            kw["dimensions"] = self._dimensions
        return kw

    def embed(self, texts: list[str]) -> list[list[float]]:
        openai = _require_openai()
        client = openai.OpenAI(**self._client_kwargs())
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = client.embeddings.create(
                input=batch, model=self._model, **self._call_kwargs()
            )
            results.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return results

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        openai = _require_openai()
        client = openai.AsyncOpenAI(**self._client_kwargs())
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await client.embeddings.create(
                input=batch, model=self._model, **self._call_kwargs()
            )
            results.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return results
