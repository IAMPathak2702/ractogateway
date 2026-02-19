"""HTML reader — uses stdlib ``html.parser`` (no extra deps)."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from ractogateway.rag._models.document import Document
from ractogateway.rag.readers.base import BaseReader

_SKIP_TAGS = {"script", "style", "head", "noscript", "nav", "footer", "header"}
_BLOCK_TAGS = {
    "p",
    "div",
    "article",
    "section",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "td",
    "th",
    "blockquote",
    "pre",
    "br",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0
        self._current_tag = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._current_tag = tag.lower()
        if self._current_tag in _SKIP_TAGS:
            self._skip_depth += 1
        if self._current_tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if t in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of blank lines
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


class HtmlReader(BaseReader):
    """Extract visible text from HTML files using the stdlib HTML parser.

    No external dependencies required.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".html", ".htm", ".xhtml"})

    def read(self, path: Path) -> Document:
        try:
            raw_html = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_html = path.read_text(encoding="latin-1")

        extractor = _TextExtractor()
        extractor.feed(raw_html)
        content = extractor.get_text()

        # Extract title if present
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", raw_html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""

        return Document(
            content=content,
            source=str(path.resolve()),
            metadata={
                "extension": path.suffix.lower(),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "title": title,
            },
        )
