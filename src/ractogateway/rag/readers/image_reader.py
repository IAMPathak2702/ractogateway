"""Image reader — uses ``Pillow`` (lazy import) to extract metadata.

Images are represented as a textual description of their EXIF/metadata,
plus an optional prompt to an LLM for visual description.  Pixel data is
not stored in the Document; use :class:`~ractogateway.prompts.engine.RactoFile`
for multimodal vision calls.

Install with:  pip install ractogateway[rag-image]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_pillow() -> Any:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Reading image files requires the 'Pillow' package. "
            "Install it with:  pip install ractogateway[rag-image]"
        ) from exc
    return Image


from ractogateway.rag._models.document import Document
from ractogateway.rag.readers.base import BaseReader

_EXIF_TAGS: dict[int, str] = {}


def _load_exif_tags() -> dict[int, str]:
    global _EXIF_TAGS  # noqa: PLW0603
    if not _EXIF_TAGS:
        try:
            from PIL.ExifTags import TAGS  # noqa: PLC0415

            _EXIF_TAGS = {k: v for k, v in TAGS.items()}
        except Exception:  # noqa: BLE001
            pass
    return _EXIF_TAGS


class ImageReader(BaseReader):
    """Extract metadata from image files and represent them as text Documents.

    The resulting ``Document.content`` is a human-readable summary of image
    properties (size, mode, format, EXIF tags).  Pass the image to a vision
    LLM separately using :class:`~ractogateway.prompts.engine.RactoFile` for
    actual visual understanding.

    Parameters
    ----------
    include_exif:
        Whether to extract and include EXIF metadata in the content.
    """

    def __init__(self, include_exif: bool = True) -> None:
        self._include_exif = include_exif

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({
            ".png", ".jpg", ".jpeg", ".webp",
            ".gif", ".bmp", ".tiff", ".tif",
        })

    def read(self, path: Path) -> Document:
        Image = _require_pillow()

        with Image.open(str(path)) as img:
            width, height = img.size
            mode = img.mode
            fmt = img.format or path.suffix.upper().lstrip(".")

            lines: list[str] = [
                f"Image: {path.name}",
                f"Format: {fmt}",
                f"Size: {width}x{height} pixels",
                f"Color mode: {mode}",
            ]

            if self._include_exif:
                exif_data = self._extract_exif(img)
                if exif_data:
                    lines.append("EXIF metadata:")
                    for key, val in exif_data.items():
                        lines.append(f"  {key}: {val}")

        content = "\n".join(lines)
        return Document(
            content=content,
            source=str(path.resolve()),
            metadata={
                "extension": path.suffix.lower(),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "format": fmt,
                "mode": mode,
            },
        )

    def _extract_exif(self, img: Any) -> dict[str, str]:
        tags = _load_exif_tags()
        result: dict[str, str] = {}
        try:
            raw_exif = img._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = tags.get(tag_id, str(tag_id))
                    if isinstance(value, bytes):
                        continue  # skip binary blobs
                    result[tag_name] = str(value)[:200]
        except Exception:  # noqa: BLE001
            pass
        return result
