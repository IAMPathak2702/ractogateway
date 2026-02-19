"""Word document reader — uses ``python-docx`` (lazy import).

Install with:  pip install ractogateway[rag-word]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_docx() -> Any:
    try:
        import docx
    except ImportError as exc:
        raise ImportError(
            "Reading Word (.docx) files requires the 'python-docx' package. "
            "Install it with:  pip install ractogateway[rag-word]"
        ) from exc
    return docx


from ractogateway.rag._models.document import Document
from ractogateway.rag.readers.base import BaseReader


class WordReader(BaseReader):
    """Extract text from Microsoft Word (.docx) files using ``python-docx``."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".docx"})

    def read(self, path: Path) -> Document:
        docx = _require_docx()
        doc = docx.Document(str(path))

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = "\t".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        full_text = "\n\n".join(paragraphs)

        core_props = doc.core_properties
        return Document(
            content=full_text,
            source=str(path.resolve()),
            metadata={
                "extension": ".docx",
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "author": core_props.author or "",
                "title": core_props.title or "",
                "paragraph_count": len(paragraphs),
            },
        )
