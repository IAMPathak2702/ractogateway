"""ProcessingPipeline — chain multiple processors sequentially."""

from __future__ import annotations

from ractogateway.rag.processors.base import BaseProcessor


class ProcessingPipeline(BaseProcessor):
    """Apply a sequence of :class:`BaseProcessor` objects to text.

    Example::

        pipeline = ProcessingPipeline([TextCleaner(), Lemmatizer()])
        processed = pipeline.process("  Hello,   worlds!  ")

    Parameters
    ----------
    processors:
        Ordered list of processors to apply.  Each processor receives the
        output of the previous one.
    """

    def __init__(self, processors: list[BaseProcessor]) -> None:
        if not processors:
            raise ValueError("ProcessingPipeline requires at least one processor.")
        self._processors = processors

    def process(self, text: str) -> str:
        result = text
        for proc in self._processors:
            result = proc.process(result)
        return result

    def __repr__(self) -> str:
        names = " → ".join(type(p).__name__ for p in self._processors)
        return f"ProcessingPipeline([{names}])"
