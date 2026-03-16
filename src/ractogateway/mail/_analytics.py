"""Email analytics engine for RactoMail."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone

from ._models import EmailAnalytics, MailMessage
from ._thread import detect_sentiment


def _ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class EmailAnalyticsEngine:
    """Compute analytics and relationship health across indexed messages."""

    def compute(
        self,
        messages: list[MailMessage],
        *,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> EmailAnalytics:
        now = datetime.now(tz=timezone.utc)
        if not messages:
            return EmailAnalytics(
                period_start=period_start or now,
                period_end=period_end or now,
            )
        sent_ats = [_ensure_utc(m.sent_at) for m in messages]
        start = _ensure_utc(period_start) if period_start else min(sent_ats)
        end = _ensure_utc(period_end) if period_end else max(sent_ats)
        filtered = [m for m in messages if start <= _ensure_utc(m.sent_at) <= end]

        sender_counter: Counter[str] = Counter(m.sender.email for m in filtered)
        top_senders = sender_counter.most_common(10)

        sentiment_dist: dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
        rel_health: dict[str, float] = {}
        for m in filtered:
            label, _ = detect_sentiment(m.combined_text())
            sentiment_dist[label] = sentiment_dist.get(label, 0) + 1
            email = m.sender.email
            delta = 0.1 if label == "positive" else (-0.1 if label == "negative" else 0.0)
            rel_health[email] = round(
                min(1.0, max(0.0, rel_health.get(email, 0.5) + delta)), 3
            )

        seen_hashes: set[str] = set()
        duplicates = 0
        for m in filtered:
            key = hashlib.sha256((m.subject + m.body_text).encode("utf-8")).hexdigest()
            if key in seen_hashes:
                duplicates += 1
            else:
                seen_hashes.add(key)

        return EmailAnalytics(
            period_start=start,
            period_end=end,
            total_emails=len(filtered),
            unique_contacts=len(sender_counter),
            sentiment_distribution=sentiment_dist,
            top_senders=top_senders,
            duplicate_count=duplicates,
            relationship_health=rel_health,
        )
