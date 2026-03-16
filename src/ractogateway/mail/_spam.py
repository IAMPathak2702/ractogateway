"""Spam and noise filtering for RactoMail."""

from __future__ import annotations

import re

from ._models import MailMessage, SpamConfig

_PROMOTIONAL_RE = re.compile(
    r"unsubscribe|email preferences|opt[\s-]out|click here to unsubscribe|"
    r"you are receiving this (email|message|newsletter)|promotional|"
    r"newsletter|special offer|limited time|coupon|discount code",
    re.IGNORECASE,
)
_AUTOMATED_RE = re.compile(
    r"no[\s-]?reply|donotreply|"
    r"automated (message|email|notification|response|reply)|"
    r"this is an automated|do not reply to this",
    re.IGNORECASE,
)


class SpamFilter:
    """Multi-layer spam and noise filter."""

    def __init__(self, config: SpamConfig) -> None:
        self._config = config

    def is_spam(self, message: MailMessage) -> bool:
        """Return True if the message should be filtered out."""
        c = self._config
        if c.use_provider_flags and message.is_spam:
            return True
        sender = message.sender.email.casefold()
        if any(sender == b.casefold() or b.casefold() in sender for b in c.sender_blacklist):
            return True
        if any(sender == w.casefold() or w.casefold() in sender for w in c.sender_whitelist):
            return False
        combined = message.combined_text()
        if any(kw.casefold() in combined.casefold() for kw in c.keyword_blacklist):
            return True
        if c.block_promotional and _PROMOTIONAL_RE.search(combined):
            return True
        if c.block_automated and _AUTOMATED_RE.search(
            message.sender.email + " " + message.body_text
        ):
            return True
        return False

    def filter_messages(self, messages: list[MailMessage]) -> list[MailMessage]:
        """Return only the messages that pass the spam filter."""
        return [m for m in messages if not self.is_spam(m)]
