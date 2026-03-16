"""Thread reconstruction and intelligence for RactoMail."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ._models import EmailThread, MailAddress, MailMessage, ThreadEmail

_QUOTE_PATTERNS = [
    re.compile(r"^>.*$", re.MULTILINE),
    re.compile(r"^On .+wrote:\s*$", re.MULTILINE),
    re.compile(r"-{3,}.*[Oo]riginal [Mm]essage.*-{3,}", re.IGNORECASE),
    re.compile(r"(?:^From:.*\n(?:Sent:|Date:).*\nTo:.*\n(?:Subject:.*)?)", re.MULTILINE),
]

_SIGNATURE_PATTERNS = [
    re.compile(r"\n--\s*\n.*$", re.DOTALL),
    re.compile(r"\n[Bb]est [Rr]egards?[,.].*$", re.DOTALL),
    re.compile(r"\n[Tt]hanks?\s*[,.].*$", re.DOTALL),
    re.compile(r"\n[Rr]egards?\s*[,.].*$", re.DOTALL),
    re.compile(r"\n[Ss]incerely[,.].*$", re.DOTALL),
    re.compile(r"\n[Ww]ith [Tt]hanks[,.].*$", re.DOTALL),
]

_COMMITMENT_KEYWORDS = [
    "will send", "will deliver", "will provide", "will get back",
    "shall send", "shall deliver", "promise", "commit",
    "by end of", "by friday", "by monday", "by tuesday", "by wednesday",
    "by thursday", "by saturday", "by sunday", "before the deadline",
    "deliver by", "send you", "get back to you", "follow up",
    "will do", "i'll", "we'll",
]

_DECISION_KEYWORDS = [
    "decided", "agreed to", "approved", "confirmed", "finalized",
    "go ahead", "proceed", "rejected", "cancelled", "accepted",
    "concluded", "resolved", "we have agreed",
]

_OPEN_ITEM_KEYWORDS = [
    "pending", "awaiting", "need to", "please clarify", "can you",
    "could you", "when will", "update on", "still waiting",
    "not yet", "outstanding",
]

_SENTIMENT_POSITIVE_WORDS = frozenset([
    "happy", "pleased", "great", "excellent", "thank", "appreciate",
    "good", "wonderful", "satisfied", "resolved", "perfect", "agree",
    "helpful", "smooth", "successful", "fantastic",
])

_SENTIMENT_NEGATIVE_WORDS = frozenset([
    "unhappy", "disappointed", "issue", "problem", "concern", "dispute",
    "error", "failed", "delay", "incorrect", "wrong", "complaint",
    "urgent", "escalate", "angry", "frustrated", "dissatisfied", "unacceptable",
])


def strip_quotes(text: str) -> str:
    """Remove quoted reply content from an email body."""
    result = text
    for pattern in _QUOTE_PATTERNS:
        result = pattern.sub("", result)
    return result.strip()


def strip_signature(text: str) -> str:
    """Remove email signature from body text."""
    result = text
    for pattern in _SIGNATURE_PATTERNS:
        result = pattern.sub("", result)
    return result.strip()


def clean_body(text: str) -> str:
    """Strip both quotes and signature from email body."""
    return strip_signature(strip_quotes(text))


def detect_sentiment(text: str) -> tuple[str, float]:
    """Simple heuristic sentiment detection. Returns (label, score -1..1)."""
    lowered = text.casefold()
    words = set(re.findall(r"[a-z]+", lowered))
    pos = len(words & _SENTIMENT_POSITIVE_WORDS)
    neg = len(words & _SENTIMENT_NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return "neutral", 0.0
    total = pos + neg
    score = (pos - neg) / total
    if score > 0.15:
        return "positive", round(score, 3)
    if score < -0.15:
        return "negative", round(score, 3)
    return "neutral", round(score, 3)


def extract_commitments_from_text(text: str) -> list[str]:
    """Extract commitment-like sentences from text."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    results: list[str] = []
    for sentence in sentences:
        lowered = sentence.casefold()
        if any(kw in lowered for kw in _COMMITMENT_KEYWORDS) and len(sentence.strip()) > 10:
            results.append(sentence.strip())
    return results


def extract_decisions_from_text(text: str) -> list[str]:
    """Extract decision-like sentences from text."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    results: list[str] = []
    for sentence in sentences:
        lowered = sentence.casefold()
        if any(kw in lowered for kw in _DECISION_KEYWORDS) and len(sentence.strip()) > 10:
            results.append(sentence.strip())
    return results


def extract_open_items_from_text(text: str) -> list[str]:
    """Extract open questions and pending items from text."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    results: list[str] = []
    for sentence in sentences:
        lowered = sentence.casefold()
        if any(kw in lowered for kw in _OPEN_ITEM_KEYWORDS) and len(sentence.strip()) > 10:
            results.append(sentence.strip())
    return results[:5]


def _dedupe_ordered(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_thread(messages: list[MailMessage], thread_id: str) -> EmailThread:
    """Reconstruct an EmailThread from a list of messages sharing thread_id."""
    thread_messages = sorted(
        [m for m in messages if m.thread_id == thread_id],
        key=lambda m: _ensure_utc(m.sent_at),
    )
    participants_map: dict[str, MailAddress] = {}
    thread_emails: list[ThreadEmail] = []
    all_decisions: list[str] = []
    all_commitments: list[str] = []
    all_open_items: list[str] = []
    sentiment_arc: list[str] = []
    response_times: list[float] = []
    prev_sent_at: datetime | None = None
    subject = thread_messages[0].subject if thread_messages else ""
    connector_id = thread_messages[0].connector_id if thread_messages else ""

    for position, message in enumerate(thread_messages, start=1):
        participants_map[message.sender.email] = message.sender
        clean_text = clean_body(message.body_text)
        all_commitments.extend(extract_commitments_from_text(clean_text))
        all_decisions.extend(extract_decisions_from_text(clean_text))
        all_open_items.extend(extract_open_items_from_text(clean_text))
        sentiment_label, _ = detect_sentiment(clean_text)
        sentiment_arc.append(sentiment_label)

        if prev_sent_at is not None:
            sent_utc = _ensure_utc(message.sent_at)
            prev_utc = _ensure_utc(prev_sent_at)
            diff_hours = (sent_utc - prev_utc).total_seconds() / 3600
            if diff_hours >= 0:
                response_times.append(round(diff_hours, 2))
        prev_sent_at = message.sent_at

        thread_emails.append(
            ThreadEmail(
                message_id=message.message_id,
                subject=message.subject,
                sender=message.sender,
                sent_at=_ensure_utc(message.sent_at),
                body_text=clean_text,
                position=position,
            )
        )

    summary_parts = [f"Thread '{subject}' — {len(thread_messages)} message(s)."]
    if all_decisions:
        summary_parts.append(f"Key decision: {all_decisions[0][:80]}")
    if all_commitments:
        summary_parts.append(f"Key commitment: {all_commitments[0][:80]}")

    return EmailThread(
        thread_id=thread_id,
        subject=subject,
        connector_id=connector_id,
        participants=list(participants_map.values()),
        messages=thread_emails,
        decisions=_dedupe_ordered(all_decisions),
        commitments=_dedupe_ordered(all_commitments),
        open_items=_dedupe_ordered(all_open_items)[:5],
        response_times_hours=response_times,
        sentiment_arc=sentiment_arc,
        summary=" ".join(summary_parts),
    )
