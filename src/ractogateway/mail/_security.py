"""Security and privacy utilities for RactoMail."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ._models import AuditLogEntry


class AuditLog:
    """In-memory audit log for mail operations."""

    def __init__(self) -> None:
        self._entries: list[AuditLogEntry] = []

    def record(
        self,
        operation: str,
        *,
        query: str | None = None,
        user: str | None = None,
        result_count: int | None = None,
    ) -> None:
        """Append an audit entry."""
        self._entries.append(
            AuditLogEntry(
                timestamp=datetime.now(tz=timezone.utc),
                operation=operation,
                query=query,
                user=user,
                result_count=result_count,
            )
        )

    def entries(self) -> list[AuditLogEntry]:
        """Return a snapshot of all audit entries."""
        return list(self._entries)

    def clear(self) -> None:
        """Wipe all audit entries."""
        self._entries.clear()

    def count(self) -> int:
        return len(self._entries)


class RightToForget:
    """Handle right-to-forget (GDPR-style) requests for email purging."""

    def __init__(self) -> None:
        self._purged: set[str] = set()

    def purge(self, email: str) -> None:
        """Register an email address for removal from the index."""
        self._purged.add(email.casefold())

    def is_purged(self, email: str) -> bool:
        """Return True if the address has been purged."""
        return email.casefold() in self._purged

    def apply_to_messages(self, messages: list[Any]) -> list[Any]:
        """Remove messages from purged senders."""
        if not self._purged:
            return messages
        try:
            from ractogateway.mail._models import MailMessage  # noqa: PLC0415
        except ImportError:
            return messages
        return [
            m
            for m in messages
            if not (
                isinstance(m, MailMessage) and m.sender.email.casefold() in self._purged
            )
        ]

    def purged_count(self) -> int:
        return len(self._purged)
