"""Plain-text reader — handles .txt, .md, .rst, .log and similar files."""

from __future__ import annotations

from pathlib import Path

from ractogateway.rag._models.document import Document
from ractogateway.rag.readers.base import BaseReader


class TextReader(BaseReader):
    """Read any UTF-8 (or latin-1 fallback) plain-text file.

    No external dependencies required.

    Parameters
    ----------
    encoding:
        Primary encoding to try.  Falls back to ``"latin-1"`` on error.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(
            {
                ".txt",
                ".md",
                ".markdown",
                ".rst",
                ".log",
                ".ini",
                ".cfg",
                ".toml",
                ".yaml",
                ".yml",
                ".json",
                ".jsonl",
                ".xml",
                ".tex",
            }
        )

    def read(self, path: Path) -> Document:
        try:
            content = path.read_text(encoding=self._encoding)
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

        return Document(
            content=content,
            source=str(path.resolve()),
            metadata={
                "extension": path.suffix.lower(),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
            },
        )
