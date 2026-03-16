"""Testing helpers for RactoMail."""

from __future__ import annotations

from collections.abc import Sequence

from ._models import MailMessage
from .connectors import BaseMailConnector


class MockMailConnector(BaseMailConnector):
    """Test connector that serves a fixed set of normalized messages."""

    provider_name = "mock"

    def __init__(
        self,
        *,
        connector_id: str = "mock",
        mailbox: str = "INBOX",
        messages: Sequence[MailMessage] | None = None,
    ) -> None:
        super().__init__(connector_id=connector_id, mailbox=mailbox, messages=messages)

    @classmethod
    def from_fixtures(
        cls,
        messages: Sequence[MailMessage],
        *,
        connector_id: str = "mock",
    ) -> MockMailConnector:
        """Construct a connector directly from normalized fixtures."""
        return cls(connector_id=connector_id, messages=messages)
