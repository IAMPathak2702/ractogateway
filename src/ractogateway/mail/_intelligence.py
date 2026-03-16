"""Email intelligence: commitments, unanswered, contact profiles, negotiation, sentiment."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timezone

from ._models import (
    Commitment,
    CommitmentStatus,
    CommunicationPattern,
    ContactProfile,
    HtmlTableData,
    MailMessage,
    NegotiationResult,
    NegotiationRound,
    SentimentPoint,
    SentimentTimeline,
    UnansweredEmail,
)
from ._thread import detect_sentiment, extract_commitments_from_text
from .search import extract_amounts


class CommitmentDetector:
    """Rule-based commitment detection from email messages."""

    def detect(self, messages: list[MailMessage]) -> list[Commitment]:
        commitments: list[Commitment] = []
        for message in messages:
            texts = extract_commitments_from_text(message.combined_text())
            for idx, commit_text in enumerate(texts):
                commitment_id = hashlib.sha256(
                    f"{message.message_id}:{idx}".encode()
                ).hexdigest()[:16]
                commitments.append(
                    Commitment(
                        commitment_id=commitment_id,
                        message_id=message.message_id,
                        thread_id=message.thread_id,
                        connector_id=message.connector_id,
                        sender=message.sender.display,
                        text=commit_text,
                        due_date=None,
                        status=CommitmentStatus.UNKNOWN,
                        confidence=0.7,
                    )
                )
        return commitments


class UnansweredDetector:
    """Detect single-message threads that have not received a reply."""

    def detect(
        self,
        messages: list[MailMessage],
        *,
        important_senders: list[str] | None = None,
        min_age_hours: float = 24.0,
    ) -> list[UnansweredEmail]:
        important = set(s.casefold() for s in (important_senders or []))
        now = datetime.now(tz=timezone.utc)
        thread_counts: Counter[str] = Counter(m.thread_id for m in messages)
        unanswered: list[UnansweredEmail] = []
        seen_threads: set[str] = set()

        for message in messages:
            if message.thread_id in seen_threads:
                continue
            seen_threads.add(message.thread_id)
            if thread_counts[message.thread_id] > 1:
                continue
            sent_utc = (
                message.sent_at.replace(tzinfo=timezone.utc)
                if message.sent_at.tzinfo is None
                else message.sent_at
            )
            age_hours = (now - sent_utc).total_seconds() / 3600
            if age_hours < min_age_hours:
                continue
            age_days = age_hours / 24
            priority = "normal"
            sender_low = message.sender.email.casefold()
            if any(s in sender_low for s in important) or age_days > 7:
                priority = "high"
            elif age_days < 2:
                priority = "low"
            unanswered.append(
                UnansweredEmail(
                    message_id=message.message_id,
                    thread_id=message.thread_id,
                    connector_id=message.connector_id,
                    subject=message.subject,
                    sender=message.sender.display,
                    sent_at=sent_utc,
                    age_days=round(age_days, 1),
                    priority=priority,
                )
            )
        unanswered.sort(key=lambda u: u.age_days, reverse=True)
        return unanswered


class ContactProfiler:
    """Build relationship intelligence profiles from email history."""

    def profile(self, messages: list[MailMessage], contact_email: str) -> ContactProfile:
        contact_messages = sorted(
            [m for m in messages if m.sender.email.casefold() == contact_email.casefold()],
            key=lambda m: m.sent_at,
        )
        if not contact_messages:
            return ContactProfile(email=contact_email)
        name = contact_messages[0].sender.name
        hours = [m.sent_at.hour for m in contact_messages]
        most_active_hour = Counter(hours).most_common(1)[0][0] if hours else None
        if len(contact_messages) >= 2:
            first_utc = (
                contact_messages[0].sent_at.replace(tzinfo=timezone.utc)
                if contact_messages[0].sent_at.tzinfo is None
                else contact_messages[0].sent_at
            )
            last_utc = (
                contact_messages[-1].sent_at.replace(tzinfo=timezone.utc)
                if contact_messages[-1].sent_at.tzinfo is None
                else contact_messages[-1].sent_at
            )
            months = max(1.0, (last_utc - first_utc).days / 30)
            freq = round(len(contact_messages) / months, 2)
        else:
            freq = float(len(contact_messages))
        sentiments = [detect_sentiment(m.combined_text())[0] for m in contact_messages]
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        if pos > neg * 2:
            trend = "positive"
        elif neg > pos * 2:
            trend = "negative"
        elif pos > 0 and neg > 0:
            trend = "mixed"
        else:
            trend = "neutral"
        health = {"positive": 0.8, "neutral": 0.5, "mixed": 0.4, "negative": 0.2}.get(trend, 0.5)
        unresolved: list[str] = []
        for m in contact_messages[-5:]:
            text = m.combined_text()
            if any(w in text.casefold() for w in ["issue", "problem", "pending", "unresolved"]):
                unresolved.append(text[:80].replace("\n", " ").strip())
        detector = CommitmentDetector()
        commitments = detector.detect(contact_messages)
        summary = (
            f"{contact_email} has sent {len(contact_messages)} email(s). "
            f"Sentiment trend: {trend}."
        )
        if unresolved:
            summary += f" {len(unresolved)} unresolved issue(s) detected."
        return ContactProfile(
            email=contact_email,
            name=name,
            total_emails=len(contact_messages),
            communication_pattern=CommunicationPattern(
                email_frequency_per_month=freq,
                most_active_hour=most_active_hour,
            ),
            sentiment_trend=trend,
            business_relationship_summary=summary,
            unresolved_issues=unresolved,
            outstanding_commitments=commitments,
            relationship_health_score=health,
        )


class NegotiationTracker:
    """Detect negotiation patterns across email threads."""

    def analyze_thread(self, messages: list[MailMessage], thread_id: str) -> NegotiationResult:
        thread_messages = sorted(
            [m for m in messages if m.thread_id == thread_id],
            key=lambda m: m.sent_at,
        )
        if not thread_messages:
            return NegotiationResult(thread_id=thread_id, subject="")
        subject = thread_messages[0].subject
        rounds: list[NegotiationRound] = []
        _NEG_KEYWORDS = {"offer", "quote", "price", "proposal", "counter", "negotiate", "bid"}
        for idx, message in enumerate(thread_messages, start=1):
            text = message.combined_text()
            amounts = extract_amounts(text)
            price = amounts[0] if amounts else None
            if price is not None or any(w in text.casefold() for w in _NEG_KEYWORDS):
                if idx == 1:
                    position = "opening"
                elif idx == len(thread_messages):
                    position = "final"
                else:
                    position = "counter"
                rounds.append(
                    NegotiationRound(
                        round_number=idx,
                        message_id=message.message_id,
                        sender=message.sender.display,
                        sent_at=message.sent_at,
                        price_mentioned=price,
                        position=position,
                    )
                )
        final_price = rounds[-1].price_mentioned if rounds else None
        if len(rounds) < 2:
            outcome = "ongoing"
        elif final_price is not None:
            outcome = "agreed"
        else:
            outcome = "no_outcome"
        pattern = f"{len(rounds)}-round negotiation" if rounds else "no_negotiation"
        return NegotiationResult(
            thread_id=thread_id,
            subject=subject,
            rounds=rounds,
            final_outcome=outcome,
            final_price=final_price,
            pattern=pattern,
        )


class SentimentAnalyzer:
    """Build sentiment timelines for individual contacts."""

    def timeline(self, messages: list[MailMessage], contact_email: str) -> SentimentTimeline:
        contact_messages = sorted(
            [m for m in messages if m.sender.email.casefold() == contact_email.casefold()],
            key=lambda m: m.sent_at,
        )
        points: list[SentimentPoint] = []
        for message in contact_messages:
            label, score = detect_sentiment(message.combined_text())
            sent_utc = (
                message.sent_at.replace(tzinfo=timezone.utc)
                if message.sent_at.tzinfo is None
                else message.sent_at
            )
            points.append(
                SentimentPoint(
                    message_id=message.message_id,
                    sender=message.sender.display,
                    sent_at=sent_utc,
                    sentiment=label,
                    score=score,
                )
            )
        if not points:
            return SentimentTimeline(contact_email=contact_email)
        sentiments = [p.sentiment for p in points]
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        overall = "positive" if pos > neg else ("negative" if neg > pos else "neutral")
        risk_level = "high" if neg > len(points) * 0.5 else ("medium" if neg > 0 else "low")
        return SentimentTimeline(
            contact_email=contact_email,
            points=points,
            overall=overall,
            risk_level=risk_level,
        )


class HtmlTableExtractor:
    """Extract structured tables from HTML email content."""

    _TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
    _ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
    _CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
    _TAG_RE = re.compile(r"<[^>]+>")

    def extract(self, html_content: str) -> list[HtmlTableData]:
        try:
            return self._extract(html_content)
        except Exception:
            return []

    def _extract(self, html: str) -> list[HtmlTableData]:
        tables: list[HtmlTableData] = []
        for table_match in self._TABLE_RE.finditer(html):
            table_html = table_match.group(1)
            rows: list[list[str]] = []
            for row_match in self._ROW_RE.finditer(table_html):
                cells = [
                    self._TAG_RE.sub("", cell_match.group(1)).strip()
                    for cell_match in self._CELL_RE.finditer(row_match.group(1))
                ]
                if cells:
                    rows.append(cells)
            if not rows:
                continue
            has_th = bool(re.search(r"<th", table_html, re.IGNORECASE))
            headers = rows[0] if has_th else []
            data_rows = rows[1:] if has_th else rows
            numeric_cols: list[int] = []
            if data_rows:
                for col_idx in range(len(data_rows[0])):
                    col_vals = [row[col_idx] for row in data_rows if col_idx < len(row)]
                    if col_vals and sum(
                        1 for v in col_vals if re.match(r"^[\d,.\s]+$", v.strip())
                    ) > len(col_vals) * 0.5:
                        numeric_cols.append(col_idx)
            all_text = " ".join(
                headers + [c for row in data_rows[:3] for c in row]
            ).casefold()
            if any(w in all_text for w in ["invoice", "bill", "gst", "tax"]):
                inferred = "invoice"
            elif any(w in all_text for w in ["balance", "debit", "credit", "account"]):
                inferred = "bank_statement"
            else:
                inferred = "generic"
            tables.append(
                HtmlTableData(
                    headers=headers,
                    rows=data_rows,
                    numeric_columns=numeric_cols,
                    inferred_type=inferred,
                )
            )
        return tables
