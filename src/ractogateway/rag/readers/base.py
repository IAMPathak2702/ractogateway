"""Abstract base class for all file readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from ractogateway.rag._models.document import Document

#: All input types accepted by :meth:`BaseReader.read`.
ReadSource = str | Path | bytes | BinaryIO


class BaseReader(ABC):
    """Read content from a file path, raw bytes, or a binary buffer.

    Concrete subclasses must implement :meth:`_read_path` and may override
    :meth:`_read_bytes` to support bytes/buffer input.  The public
    :meth:`read` method handles all type coercion automatically.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        """Lower-case extensions (with dot) this reader handles, e.g. ``{".pdf"}``."""

    def read(self, source: str | Path | bytes | BinaryIO) -> Document:
        """Load *source* and return its content as a :class:`Document`.

        Parameters
        ----------
        source:
            ``str`` or ``Path``
                File path read from disk.  Both absolute and relative paths
                are accepted.
            ``bytes``
                Raw file bytes.  ``Document.source`` is set to ``"<bytes>"``.
            binary file-like object
                Any object with a ``.read() -> bytes`` method — e.g.
                ``io.BytesIO``, an open binary file handle, a network stream.
                ``Document.source`` is set to ``"<buffer>"``.
        """
        if isinstance(source, (str, Path)):
            return self._read_path(Path(source))
        if isinstance(source, bytes):
            return self._read_bytes(source, source_label="<bytes>")
        # File-like / buffer — anything with .read()
        return self._read_bytes(source.read(), source_label="<buffer>")

    @abstractmethod
    def _read_path(self, path: Path) -> Document:
        """Read from *path* on disk.

        Always receives a fully-constructed :class:`~pathlib.Path`; no need
        to coerce inside implementations.
        """

    def _read_bytes(self, data: bytes, *, source_label: str = "<bytes>") -> Document:
        """Read from raw *data* bytes.

        The default implementation raises :class:`NotImplementedError`.
        Subclasses that wish to support bytes / buffer input should override
        this method.

        Parameters
        ----------
        data:
            Raw file bytes.
        source_label:
            Value used as ``Document.source``
            (``"<bytes>"`` or ``"<buffer>"``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support bytes/buffer input. "
            "Pass a file path (str or Path) instead."
        )
