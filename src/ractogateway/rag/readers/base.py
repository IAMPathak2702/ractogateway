"""Abstract base class for all file readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ractogateway.rag._models.document import Document


class BaseReader(ABC):
    """Read a file from disk and return a :class:`Document`.

    Concrete subclasses implement :meth:`read` and declare which file
    extensions they handle via :attr:`supported_extensions`.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        """Lower-case extensions (with dot) this reader handles, e.g. ``{".pdf"}``."""

    @abstractmethod
    def read(self, path: Path) -> Document:
        """Load *path* and return its text content as a :class:`Document`.

        Parameters
        ----------
        path:
            Absolute path to the file to read.

        Returns
        -------
        Document
            A Document whose ``source`` is set to ``str(path)`` and whose
            ``content`` is the full extracted text.
        """
