"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Embed a list of texts into dense float vectors.

    All embedders implement both sync :meth:`embed` and async :meth:`aembed`
    variants.  The dimension of returned vectors is declared via the
    :attr:`dimension` property (``-1`` if unknown until the first call).
    """

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors.

        Returns ``-1`` if not known until after the first call.
        """
        return -1

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* synchronously.

        Parameters
        ----------
        texts:
            Non-empty list of strings to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text, in the same order.
        """

    @abstractmethod
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Async variant of :meth:`embed`."""
