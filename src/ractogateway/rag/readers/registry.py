"""FileReaderRegistry — auto-detects the right reader for any file extension."""

from __future__ import annotations

from pathlib import Path

from ractogateway.rag._models.document import Document
from ractogateway.rag.readers.base import BaseReader
from ractogateway.rag.readers.html_reader import HtmlReader
from ractogateway.rag.readers.image_reader import ImageReader
from ractogateway.rag.readers.pdf_reader import PdfReader
from ractogateway.rag.readers.spreadsheet_reader import SpreadsheetReader
from ractogateway.rag.readers.text_reader import TextReader
from ractogateway.rag.readers.word_reader import WordReader


def _default_readers() -> list[BaseReader]:
    return [
        TextReader(),
        PdfReader(),
        WordReader(),
        SpreadsheetReader(),
        ImageReader(),
        HtmlReader(),
    ]


class FileReaderRegistry:
    """Registry that maps file extensions to :class:`BaseReader` instances.

    By default all built-in readers are registered.  You can add custom
    readers with :meth:`register`.

    Example::

        registry = FileReaderRegistry()
        doc = registry.read("report.pdf")
    """

    def __init__(self, readers: list[BaseReader] | None = None) -> None:
        self._map: dict[str, BaseReader] = {}
        for reader in readers or _default_readers():
            self.register(reader)

    def register(self, reader: BaseReader) -> None:
        """Add *reader* to the registry for all its supported extensions."""
        for ext in reader.supported_extensions:
            self._map[ext.lower()] = reader

    def get_reader(self, path: str | Path) -> BaseReader:
        """Return the reader for *path*'s extension.

        Raises
        ------
        ValueError
            If no reader supports the file's extension.
        """
        ext = Path(path).suffix.lower()
        reader = self._map.get(ext)
        if reader is None:
            supported = sorted(self._map.keys())
            raise ValueError(
                f"No reader registered for extension '{ext}'. Supported extensions: {supported}"
            )
        return reader

    def read(self, path: str | Path) -> Document:
        """Convenience method: detect reader and return a :class:`Document`."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return self.get_reader(p).read(p)

    @property
    def supported_extensions(self) -> frozenset[str]:
        """All extensions currently registered."""
        return frozenset(self._map.keys())
