"""PDF reader — uses ``pypdf`` (lazy import).

Install with:  pip install ractogateway[rag-pdf]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_pypdf() -> Any:
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError(
            "Reading PDF files requires the 'pypdf' package. "
            "Install it with:  pip install ractogateway[rag-pdf]"
        ) from exc
    return pypdf


from ractogateway.rag._models.document import Document
from ractogateway.rag.readers.base import BaseReader


class PdfReader(BaseReader):
    """Extract text from PDF files using ``pypdf``.

    Parameters
    ----------
    extract_images:
        Reserved for future use — image extraction is not yet supported.
    """

    def __init__(self, extract_images: bool = False) -> None:
        self._extract_images = extract_images

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def read(self, path: Path) -> Document:
        pypdf = _require_pypdf()

        pages_text: list[str] = []
        page_map: list[tuple[int, str]] = []

        with pypdf.PdfReader(str(path)) as reader:
            total_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages_text.append(text)
                page_map.append((i + 1, text))

        full_text = "\n\n".join(t for t in pages_text if t.strip())

        return Document(
            content=full_text,
            source=str(path.resolve()),
            metadata={
                "extension": ".pdf",
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "total_pages": total_pages,
                "page_map": page_map,
            },
        )
