"""Text cleaning processor — no extra dependencies."""

from __future__ import annotations

import re
import unicodedata

from ractogateway.rag.processors.base import BaseProcessor

# Control characters (except newline/tab)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Multiple blank lines → double newline
_BLANK_LINES_RE = re.compile(r"\n{3,}")
# Multiple spaces (not newlines) → single space
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
# Residual HTML tags
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class TextCleaner(BaseProcessor):
    """Normalise text for embedding and retrieval.

    Steps applied (all optional via constructor flags):

    1. Unicode normalisation (NFC)
    2. Strip residual HTML tags
    3. Remove control characters
    4. Collapse multiple spaces to one
    5. Collapse runs of blank lines to at most two newlines
    6. Strip leading/trailing whitespace

    Parameters
    ----------
    normalize_unicode:
        Apply ``unicodedata.normalize("NFC", text)``.
    strip_html:
        Remove ``<tag>`` patterns.
    strip_control_chars:
        Remove non-printable control characters.
    collapse_whitespace:
        Collapse sequences of spaces/tabs to a single space.
    collapse_blank_lines:
        Collapse 3+ consecutive newlines to 2.
    """

    def __init__(
        self,
        normalize_unicode: bool = True,
        strip_html: bool = True,
        strip_control_chars: bool = True,
        collapse_whitespace: bool = True,
        collapse_blank_lines: bool = True,
    ) -> None:
        self.normalize_unicode = normalize_unicode
        self.strip_html = strip_html
        self.strip_control_chars = strip_control_chars
        self.collapse_whitespace = collapse_whitespace
        self.collapse_blank_lines = collapse_blank_lines

    def process(self, text: str) -> str:
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)
        if self.strip_html:
            text = _HTML_TAG_RE.sub(" ", text)
        if self.strip_control_chars:
            text = _CONTROL_RE.sub("", text)
        if self.collapse_whitespace:
            text = _MULTI_SPACE_RE.sub(" ", text)
        if self.collapse_blank_lines:
            text = _BLANK_LINES_RE.sub("\n\n", text)
        return text.strip()
