"""Connector primitives for the RactoMail foundation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime

from ._models import MailMessage


class BaseMailConnector:
    """Read-only connector that returns normalized messages."""

    provider_name = "base"

    def __init__(
        self,
        *,
        connector_id: str,
        mailbox: str = "INBOX",
        messages: Sequence[MailMessage] | None = None,
    ) -> None:
        self.connector_id = connector_id
        self.mailbox = mailbox
        self._messages = list(messages or [])

    @property
    def read_only(self) -> bool:
        """All built-in connectors are read-only by design."""
        return True

    def set_messages(self, messages: Sequence[MailMessage]) -> None:
        """Replace the fixture-backed message set."""
        self._messages = list(messages)

    def fetch_messages(self, *, since: datetime | None = None) -> list[MailMessage]:
        """Return messages filtered by ``since`` if supplied."""
        if since is None:
            return list(self._messages)
        return [message for message in self._messages if message.sent_at >= since]

    async def afetch_messages(self, *, since: datetime | None = None) -> list[MailMessage]:
        """Async wrapper over :meth:`fetch_messages`."""
        return await asyncio.to_thread(self.fetch_messages, since=since)


class GmailConnector(BaseMailConnector):
    """Fixture-backed Gmail connector."""

    provider_name = "gmail"

    def __init__(
        self,
        *,
        connector_id: str = "gmail",
        mailbox: str = "INBOX",
        credentials_file: str | None = None,
        token_file: str | None = None,
        labels: Sequence[str] | None = None,
        messages: Sequence[MailMessage] | None = None,
    ) -> None:
        super().__init__(connector_id=connector_id, mailbox=mailbox, messages=messages)
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.labels = list(labels or ["INBOX", "SENT"])


class OutlookConnector(BaseMailConnector):
    """Fixture-backed Outlook / Office 365 connector."""

    provider_name = "outlook"

    def __init__(
        self,
        *,
        connector_id: str = "outlook",
        mailbox: str = "INBOX",
        client_id: str | None = None,
        client_secret: str | None = None,
        tenant_id: str | None = None,
        folders: Sequence[str] | None = None,
        messages: Sequence[MailMessage] | None = None,
    ) -> None:
        super().__init__(connector_id=connector_id, mailbox=mailbox, messages=messages)
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.folders = list(folders or ["Inbox", "Sent Items"])


class ExchangeConnector(BaseMailConnector):
    """Fixture-backed Exchange connector."""

    provider_name = "exchange"

    def __init__(
        self,
        *,
        connector_id: str = "exchange",
        mailbox: str = "INBOX",
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        messages: Sequence[MailMessage] | None = None,
    ) -> None:
        super().__init__(connector_id=connector_id, mailbox=mailbox, messages=messages)
        self.base_url = base_url
        self.username = username
        self.password = password


class IMAPConnector(BaseMailConnector):
    """Fixture-backed IMAP connector that also covers GoDaddy-style hosts."""

    provider_name = "imap"

    def __init__(
        self,
        *,
        connector_id: str = "imap",
        mailbox: str = "INBOX",
        host: str = "localhost",
        port: int = 993,
        username: str | None = None,
        password: str | None = None,
        use_ssl: bool = True,
        messages: Sequence[MailMessage] | None = None,
    ) -> None:
        super().__init__(connector_id=connector_id, mailbox=mailbox, messages=messages)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_ssl = use_ssl


class GoDaddyConnector(IMAPConnector):
    """Convenience IMAP connector preconfigured for GoDaddy."""

    provider_name = "godaddy"

    def __init__(
        self,
        *,
        connector_id: str = "godaddy",
        mailbox: str = "INBOX",
        username: str | None = None,
        password: str | None = None,
        messages: Sequence[MailMessage] | None = None,
    ) -> None:
        super().__init__(
            connector_id=connector_id,
            mailbox=mailbox,
            host="imap.secureserver.net",
            port=993,
            username=username,
            password=password,
            use_ssl=True,
            messages=messages,
        )


class ZohoConnector(BaseMailConnector):
    """Fixture-backed Zoho connector."""

    provider_name = "zoho"

    def __init__(
        self,
        *,
        connector_id: str = "zoho",
        mailbox: str = "INBOX",
        client_id: str | None = None,
        client_secret: str | None = None,
        organization_id: str | None = None,
        messages: Sequence[MailMessage] | None = None,
    ) -> None:
        super().__init__(connector_id=connector_id, mailbox=mailbox, messages=messages)
        self.client_id = client_id
        self.client_secret = client_secret
        self.organization_id = organization_id


class YahooConnector(BaseMailConnector):
    """Fixture-backed Yahoo connector."""

    provider_name = "yahoo"

    def __init__(
        self,
        *,
        connector_id: str = "yahoo",
        mailbox: str = "INBOX",
        username: str | None = None,
        app_password: str | None = None,
        messages: Sequence[MailMessage] | None = None,
    ) -> None:
        super().__init__(connector_id=connector_id, mailbox=mailbox, messages=messages)
        self.username = username
        self.app_password = app_password
