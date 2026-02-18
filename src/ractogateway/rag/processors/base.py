"""Abstract base class for text processors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """Transform a text string and return the processed result.

    Processors are applied to chunk content *before* embedding.  They can
    normalise whitespace, lemmatize tokens, remove stop words, etc.

    Chain multiple processors with
    :class:`~ractogateway.rag.processors.pipeline.ProcessingPipeline`.
    """

    @abstractmethod
    def process(self, text: str) -> str:
        """Process *text* and return the transformed string.

        Parameters
        ----------
        text:
            Input text (chunk content or raw document content).

        Returns
        -------
        str
            Processed text.  Must be a non-empty string when input is non-empty.
        """
