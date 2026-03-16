"""Unified mail intelligence kit for RactoMail."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time as _time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from ._models import (
    AmountFilter,
    AttachmentRef,
    AuditLogEntry,
    Commitment,
    ContactProfile,
    DateRange,
    DailyBriefing,
    DeltaCheckpoint,
    EmailAnalytics,
    EmailRef,
    EmailSearchResult,
    EmailThread,
    EntityResult,
    EvidenceItem,
    FacetedResult,
    FuzzyMatch,
    FuzzySearch,
    HtmlTableData,
    LanguageConfig,
    MailActionSuggestion,
    MailFacets,
    MailMessage,
    MailPrivacyConfig,
    MailQueryConfig,
    MailResponse,
    MailStats,
    NegotiationResult,
    RegexMatch,
    SavedSearch,
    SearchMode,
    SecurityConfig,
    SentimentTimeline,
    SpamConfig,
    SyncMode,
    SyncStatus,
    UnansweredEmail,
)
from .connectors import BaseMailConnector
from .flow import DAGRunStore, PythonOperator, RactoDAG
from .memory import ConversationSession, SessionStore
from .search import (
    IndianCalendar,
    evaluate_boolean_query,
    extract_amounts,
    find_fuzzy_matches,
    normalize_search_text,
    tokenize,
)

_STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "any",
    "before",
    "did",
    "emails",
    "find",
    "from",
    "in",
    "is",
    "me",
    "of",
    "only",
    "show",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
}


class RactoMailKit:
    """Unified email intelligence entrypoint supporting all providers and features."""

    def __init__(
        self,
        *,
        kit: Any | None = None,
        connectors: Sequence[BaseMailConnector] | None = None,
        privacy: MailPrivacyConfig | None = None,
        session_store: SessionStore | None = None,
        run_store: DAGRunStore | None = None,
        spam_config: SpamConfig | None = None,
        security_config: SecurityConfig | None = None,
        language_config: LanguageConfig | None = None,
    ) -> None:
        self._kit = kit
        self._connectors = list(connectors or [])
        self._privacy = privacy or MailPrivacyConfig()
        self._session_store = session_store or SessionStore()
        self._run_store = run_store or DAGRunStore()
        self._calendar = IndianCalendar()
        self._messages: list[MailMessage] = []
        self._saved_searches: dict[str, SavedSearch] = {}
        self._query_history: list[MailQueryConfig] = []

        from ._spam import SpamFilter  # noqa: PLC0415
        from ._security import AuditLog, RightToForget  # noqa: PLC0415

        self._spam_filter = SpamFilter(spam_config) if spam_config is not None else None
        self._security = security_config or SecurityConfig()
        self._language = language_config or LanguageConfig()
        self._audit_log: AuditLog = AuditLog()
        self._right_to_forget: RightToForget = RightToForget()
        self._spam_filtered_count: int = 0
        # Apply right-to-forget from security config
        for email in self._security.right_to_forget_emails:
            self._right_to_forget.purge(email)

    def start_session(
        self,
        *,
        session_id: str,
        user: str,
        context: str | None = None,
    ) -> ConversationSession:
        """Start or restore a multi-turn conversation session."""
        existing = self._session_store.get(session_id)
        if existing is not None:
            return existing
        session = ConversationSession(
            session_id=session_id,
            user=user,
            context=context,
            mail=self,
            store=self._session_store,
        )
        self._session_store.save(session)
        return session

    def save_search(self, name: str, query: str, **kwargs: Any) -> SavedSearch:
        """Persist a query config for later reuse."""
        config = self._build_query_config(query, **kwargs)
        saved = SavedSearch(
            name=name,
            config=config,
            created_at=datetime.now(tz=timezone.utc),
        )
        self._saved_searches[name] = saved
        return saved

    def run_saved_search(self, name: str) -> MailResponse:
        """Execute a previously saved search as an ``ask`` flow."""
        saved = self._saved_searches[name]
        return self.ask(saved.config.query, **saved.config.model_dump(exclude={"query"}))

    def sync(
        self,
        *,
        mode: SyncMode | str = SyncMode.PASSIVE,
        since: datetime | None = None,
        use_flow: bool = True,
    ) -> SyncStatus:
        """Fetch and index messages from all connectors."""
        sync_mode = SyncMode(mode)
        if use_flow and self._connectors:
            dag = self._build_sync_dag(since=since)
            run = dag.run()
            messages: list[MailMessage] = []
            for task_run in run.task_runs.values():
                output = task_run.output
                if isinstance(output, list):
                    messages.extend(
                        message for message in output if isinstance(message, MailMessage)
                    )
            return self._materialize_index(
                messages,
                mode=sync_mode,
                used_flow=True,
                dag_run_id=run.run_id,
            )

        messages = self._fetch_messages_sequential(since=since)
        return self._materialize_index(messages, mode=sync_mode, used_flow=False, dag_run_id=None)

    async def async_sync(
        self,
        *,
        mode: SyncMode | str = SyncMode.PASSIVE,
        since: datetime | None = None,
    ) -> SyncStatus:
        """Async sync variant that fetches all connectors concurrently."""
        sync_mode = SyncMode(mode)
        fetched = await asyncio.gather(
            *[connector.afetch_messages(since=since) for connector in self._connectors]
        )
        messages = [message for batch in fetched for message in batch]
        return self._materialize_index(messages, mode=sync_mode, used_flow=True, dag_run_id=None)

    def search(self, query: str, **kwargs: Any) -> list[EmailSearchResult]:
        """Run ranked search without generating a narrative answer."""
        config = self._build_query_config(query, **kwargs)
        results, _, _ = self._execute_search(config)
        return results

    def faceted_search(self, query: str, **kwargs: Any) -> FacetedResult:
        """Return search hits plus connector/sender/label facet counts."""
        config = self._build_query_config(query, top_k=25, **kwargs)
        results, _, _ = self._execute_search(config)
        connector_counts = Counter(result.connector_id for result in results)
        sender_counts = Counter(result.sender for result in results)
        label_counts: Counter[str] = Counter()
        message_map = {message.message_id: message for message in self._messages}
        for result in results:
            message = message_map.get(result.message_id)
            if message is not None:
                label_counts.update(message.labels)
        return FacetedResult(
            query=query,
            total_matches=len(results),
            connector_counts=dict(connector_counts),
            sender_counts=dict(sender_counts),
            label_counts=dict(label_counts),
            emails=results,
        )

    def entity_search(self, entity: str, *, top_k: int = 10) -> EntityResult:
        """Return the strongest hits for one entity string."""
        results = self.search(entity, top_k=top_k)
        return EntityResult(entity=entity, total_matches=len(results), emails=results)

    def ask(self, query: str, **kwargs: Any) -> MailResponse:
        """Search across indexed mailboxes and return a grounded answer."""
        t0 = _time.monotonic()
        config = self._build_query_config(query, **kwargs)
        self._query_history.append(config)
        results, fuzzy_matches, resolved_date_range = self._execute_search(config)
        references = self._build_references(results, config)
        answer = self._build_answer(query, results, references, config.verbatim)
        evidence = [
            EvidenceItem(
                summary=f"Matched {reference.subject} from {reference.sender}",
                reference=reference,
            )
            for reference in references
        ]
        actions = self._build_actions(results)
        query_time_ms = round((_time.monotonic() - t0) * 1000, 2)

        # Compute confidence score from top result score
        confidence = round(results[0].score / 10.0, 3) if results else 0.0
        confidence = min(1.0, max(0.0, confidence))

        if self._security.audit_logging:
            self._audit_log.record(
                "ask",
                query=query,
                result_count=len(results),
            )

        return MailResponse(
            query=query,
            answer=answer,
            references=references,
            evidence_chain=evidence,
            search_results=results,
            fuzzy_matches=fuzzy_matches,
            actions=actions,
            total_matches=len(results),
            search_mode=config.search_mode,
            resolved_date_range=resolved_date_range,
            confidence_score=confidence,
            sources_searched=len(self._connectors) or 1,
            query_time_ms=query_time_ms,
            missing_data_explanation="" if results else f"No emails matched '{query}'.",
        )

    async def aask(self, query: str, **kwargs: Any) -> MailResponse:
        """Async variant of :meth:`ask`."""
        return await asyncio.to_thread(self.ask, query, **kwargs)

    def stats(self) -> MailStats:
        """Return small operational counters."""
        return MailStats(
            indexed_messages=len(self._messages),
            indexed_threads=len({message.thread_id for message in self._messages}),
            saved_searches=len(self._saved_searches),
            sessions=self._session_store.count(),
            spam_filtered_count=self._spam_filtered_count,
            audit_log_entries=self._audit_log.count(),
        )

    def history(self) -> list[MailQueryConfig]:
        """Return the recorded query configs."""
        return list(self._query_history)

    # ── New public intelligence methods ─────────────────────────────────────

    def get_thread(self, thread_id: str) -> EmailThread:
        """Reconstruct an EmailThread from indexed messages."""
        from ._thread import build_thread  # noqa: PLC0415
        return build_thread(self._messages, thread_id)

    def get_commitments(
        self,
        *,
        connector_id: str | None = None,
        thread_id: str | None = None,
    ) -> list[Commitment]:
        """Detect commitments across indexed messages."""
        from ._intelligence import CommitmentDetector  # noqa: PLC0415
        messages = self._messages
        if connector_id is not None:
            messages = [m for m in messages if m.connector_id == connector_id]
        if thread_id is not None:
            messages = [m for m in messages if m.thread_id == thread_id]
        return CommitmentDetector().detect(messages)

    def get_unanswered(
        self,
        *,
        important_senders: list[str] | None = None,
        min_age_hours: float = 24.0,
    ) -> list[UnansweredEmail]:
        """Return emails that appear to be awaiting a reply."""
        from ._intelligence import UnansweredDetector  # noqa: PLC0415
        return UnansweredDetector().detect(
            self._messages,
            important_senders=important_senders,
            min_age_hours=min_age_hours,
        )

    def contact_profile(self, email: str) -> ContactProfile:
        """Build a relationship intelligence profile for a contact."""
        from ._intelligence import ContactProfiler  # noqa: PLC0415
        return ContactProfiler().profile(self._messages, email)

    def get_negotiation(self, thread_id: str) -> NegotiationResult:
        """Analyze negotiation patterns in a specific thread."""
        from ._intelligence import NegotiationTracker  # noqa: PLC0415
        return NegotiationTracker().analyze_thread(self._messages, thread_id)

    def sentiment_timeline(self, contact_email: str) -> SentimentTimeline:
        """Build a sentiment timeline for a specific contact."""
        from ._intelligence import SentimentAnalyzer  # noqa: PLC0415
        return SentimentAnalyzer().timeline(self._messages, contact_email)

    def analytics(
        self,
        *,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> EmailAnalytics:
        """Compute email analytics for the indexed messages."""
        from ._analytics import EmailAnalyticsEngine  # noqa: PLC0415
        return EmailAnalyticsEngine().compute(
            self._messages,
            period_start=period_start,
            period_end=period_end,
        )

    def daily_briefing(self) -> DailyBriefing:
        """Generate a daily intelligence briefing."""
        from ._intelligence import CommitmentDetector, UnansweredDetector  # noqa: PLC0415
        from ._thread import build_thread  # noqa: PLC0415
        commitments = CommitmentDetector().detect(self._messages)
        unanswered = UnansweredDetector().detect(self._messages, min_age_hours=4.0)
        # Top 3 threads by message count
        thread_counts: Counter[str] = Counter(m.thread_id for m in self._messages)
        top_thread_ids = [tid for tid, _ in thread_counts.most_common(3)]
        top_threads = [build_thread(self._messages, tid) for tid in top_thread_ids]
        summary_parts = [
            f"Total indexed emails: {len(self._messages)}.",
            f"Pending commitments: {len(commitments)}.",
            f"Unanswered emails: {len(unanswered)}.",
        ]
        return DailyBriefing(
            generated_at=datetime.now(tz=timezone.utc),
            total_emails=len(self._messages),
            pending_commitments=commitments[:10],
            unanswered_emails=unanswered[:10],
            top_threads=top_threads,
            summary=" ".join(summary_parts),
        )

    def forget(self, email: str) -> int:
        """Apply right-to-forget: remove all messages from this sender."""
        self._right_to_forget.purge(email)
        before = len(self._messages)
        self._messages = self._right_to_forget.apply_to_messages(self._messages)
        removed = before - len(self._messages)
        if self._security.audit_logging:
            self._audit_log.record("forget", query=email, result_count=removed)
        return removed

    def audit_log(self) -> list[AuditLogEntry]:
        """Return audit log entries (requires security_config.audit_logging=True)."""
        return self._audit_log.entries()

    def extract_html_tables(self, message_id: str) -> list[HtmlTableData]:
        """Extract HTML tables from a specific message's body."""
        from ._intelligence import HtmlTableExtractor  # noqa: PLC0415
        message_map = {m.message_id: m for m in self._messages}
        message = message_map.get(message_id)
        if message is None:
            return []
        return HtmlTableExtractor().extract(message.body_text)

    def delta_checkpoints(self) -> list[DeltaCheckpoint]:
        """Return delta sync checkpoints for all connectors."""
        return [
            DeltaCheckpoint(
                connector_id=connector.connector_id,
                delta_token=f"delta_{connector.connector_id}",
                last_synced_at=datetime.now(tz=timezone.utc),
            )
            for connector in self._connectors
        ]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_sync_dag(self, *, since: datetime | None) -> RactoDAG:
        dag = RactoDAG(
            dag_id="ractomail_sync",
            max_parallel_tasks=max(1, len(self._connectors)),
            run_store=self._run_store,
        )
        for connector in self._connectors:
            dag.add_task(
                PythonOperator(
                    task_id=f"fetch_{connector.connector_id}",
                    python_callable=lambda _context, connector=connector: connector.fetch_messages(
                        since=since
                    ),
                )
            )
        return dag

    def _fetch_messages_sequential(self, *, since: datetime | None) -> list[MailMessage]:
        messages: list[MailMessage] = []
        for connector in self._connectors:
            messages.extend(connector.fetch_messages(since=since))
        return messages

    def _materialize_index(
        self,
        messages: Sequence[MailMessage],
        *,
        mode: SyncMode,
        used_flow: bool,
        dag_run_id: str | None,
    ) -> SyncStatus:
        deduplicated_count = 0
        seen: set[str] = set()
        indexed: list[MailMessage] = []
        connector_counts: Counter[str] = Counter()
        for message in sorted(messages, key=lambda item: _ensure_utc(item.sent_at)):
            dedup_key = self._dedup_key(message)
            if dedup_key in seen:
                deduplicated_count += 1
                continue
            seen.add(dedup_key)
            indexed.append(message)
            connector_counts[message.connector_id] += 1

        # Apply right-to-forget
        indexed = self._right_to_forget.apply_to_messages(indexed)

        # Apply spam filter
        spam_filtered = 0
        if self._spam_filter is not None:
            original_count = len(indexed)
            indexed = self._spam_filter.filter_messages(indexed)
            spam_filtered = original_count - len(indexed)
        self._spam_filtered_count += spam_filtered

        # Apply security exclusions
        if self._security.excluded_senders:
            excl = {s.casefold() for s in self._security.excluded_senders}
            indexed = [m for m in indexed if m.sender.email.casefold() not in excl]
        if self._security.excluded_folders:
            excl_folders = {f.casefold() for f in self._security.excluded_folders}
            indexed = [m for m in indexed if m.mailbox.casefold() not in excl_folders]
        if self._security.excluded_subjects:
            for pattern in self._security.excluded_subjects:
                indexed = [
                    m for m in indexed if pattern.casefold() not in m.subject.casefold()
                ]

        self._messages = indexed
        return SyncStatus(
            mode=mode,
            indexed_count=len(indexed),
            thread_count=len({message.thread_id for message in indexed}),
            connector_counts=dict(connector_counts),
            deduplicated_count=deduplicated_count,
            used_flow=used_flow,
            dag_run_id=dag_run_id,
        )

    def _dedup_key(self, message: MailMessage) -> str:
        payload = "|".join(
            [
                message.message_id,
                message.connector_id,
                message.subject,
                normalize_search_text(message.body_text),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_query_config(self, query: str, **kwargs: Any) -> MailQueryConfig:
        facets = kwargs.pop("facets", None)
        if facets is None:
            facets_obj = MailFacets()
        elif isinstance(facets, MailFacets):
            facets_obj = facets
        else:
            facets_obj = MailFacets(**facets)

        fuzzy = kwargs.pop("fuzzy", None)
        if fuzzy is None:
            fuzzy_obj: FuzzySearch | None = None
        elif isinstance(fuzzy, FuzzySearch):
            fuzzy_obj = fuzzy
        else:
            fuzzy_obj = FuzzySearch(**fuzzy)

        amount_filter = kwargs.pop("amount_filter", None)
        if amount_filter is None:
            amount_filter_obj: AmountFilter | None = None
        elif isinstance(amount_filter, AmountFilter):
            amount_filter_obj = amount_filter
        else:
            amount_filter_obj = AmountFilter(**amount_filter)

        return MailQueryConfig(
            query=query,
            search_mode=SearchMode(kwargs.pop("search_mode", SearchMode.HYBRID)),
            top_k=int(kwargs.pop("top_k", 5)),
            max_references=int(kwargs.pop("max_references", 5)),
            verbatim=bool(kwargs.pop("verbatim", False)),
            fuzzy=fuzzy_obj,
            regex_patterns=dict(kwargs.pop("regex_patterns", {})),
            regex_match_mode=str(kwargs.pop("regex_match_mode", "any")),
            boolean_query=kwargs.pop("boolean_query", None),
            amount_filter=amount_filter_obj,
            facets=facets_obj,
        )

    def _execute_search(
        self,
        config: MailQueryConfig,
    ) -> tuple[list[EmailSearchResult], list[FuzzyMatch], DateRange | None]:
        resolved_facets = config.facets.model_copy(deep=True)
        resolved_date_range = self._calendar.resolve(config.query)
        if resolved_date_range is not None and resolved_facets.date_from is None:
            resolved_facets.date_from = resolved_date_range.start
            resolved_facets.date_to = resolved_date_range.end

        query_terms = [term for term in tokenize(config.query) if term not in _STOPWORDS]
        aggregated_fuzzy: dict[tuple[str, str, int], FuzzyMatch] = {}
        results: list[EmailSearchResult] = []

        for message in self._messages:
            if not self._matches_facets(message, resolved_facets):
                continue

            combined_text = message.combined_text()
            regex_matches = self._collect_regex_matches(combined_text, config.regex_patterns)
            if config.regex_patterns and not self._regex_matches_mode(
                regex_matches,
                config.regex_patterns,
                config.regex_match_mode,
            ):
                continue

            if config.boolean_query and not evaluate_boolean_query(
                combined_text,
                config.boolean_query,
            ):
                continue

            matched_amounts = extract_amounts(combined_text)
            if config.amount_filter and not self._matches_amount_filter(
                matched_amounts,
                config.amount_filter,
            ):
                continue

            fuzzy_matches = (
                find_fuzzy_matches(query_terms, tokenize(combined_text), config.fuzzy)
                if config.fuzzy is not None
                else []
            )
            for match in fuzzy_matches:
                aggregated_fuzzy[(match.original, match.matched_to, match.distance)] = match

            score = self._score_message(
                query=config.query,
                query_terms=query_terms,
                combined_text=combined_text,
                fuzzy_matches=fuzzy_matches,
                mode=config.search_mode,
            )
            if score <= 0.0 and not regex_matches and not matched_amounts:
                continue

            results.append(
                EmailSearchResult(
                    message_id=message.message_id,
                    thread_id=message.thread_id,
                    connector_id=message.connector_id,
                    mailbox=message.mailbox,
                    subject=message.subject,
                    sender=message.sender.display,
                    sent_at=_ensure_utc(message.sent_at),
                    score=round(score, 3),
                    snippet=self._make_snippet(combined_text, query_terms),
                    regex_matches=regex_matches,
                    matched_amounts=matched_amounts,
                )
            )

        results.sort(key=lambda item: (item.score, item.sent_at), reverse=True)
        return results[: config.top_k], list(aggregated_fuzzy.values()), resolved_date_range

    def _matches_facets(self, message: MailMessage, facets: MailFacets) -> bool:
        message_dt = _ensure_utc(message.sent_at)
        connector_ok = (
            facets.connectors is None or message.connector_id in facets.connectors
        )
        sender_ok = facets.senders is None or message.sender.email in facets.senders
        labels_ok = (
            facets.labels is None or bool(set(message.labels).intersection(facets.labels))
        )
        spam_ok = facets.include_spam or not message.is_spam
        date_from_ok = (
            facets.date_from is None or message_dt >= _ensure_utc(facets.date_from)
        )
        date_to_ok = facets.date_to is None or message_dt <= _ensure_utc(facets.date_to)
        message_ids_ok = (
            facets.message_ids is None or message.message_id in facets.message_ids
        )
        return all(
            (
                connector_ok,
                sender_ok,
                labels_ok,
                spam_ok,
                date_from_ok,
                date_to_ok,
                message_ids_ok,
            )
        )

    def _collect_regex_matches(
        self,
        text: str,
        regex_patterns: dict[str, str],
    ) -> list[RegexMatch]:
        matches: list[RegexMatch] = []
        for pattern_name, pattern in regex_patterns.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                matches.append(
                    RegexMatch(
                        pattern_name=pattern_name,
                        matched_text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return matches

    def _regex_matches_mode(
        self,
        regex_matches: list[RegexMatch],
        patterns: dict[str, str],
        mode: str,
    ) -> bool:
        matched_names = {match.pattern_name for match in regex_matches}
        if mode.casefold() == "all":
            return set(patterns).issubset(matched_names)
        return bool(regex_matches)

    def _matches_amount_filter(
        self,
        amounts: list[float],
        amount_filter: AmountFilter,
    ) -> bool:
        if not amounts:
            return False
        for amount in amounts:
            if amount_filter.min_amount is not None and amount < amount_filter.min_amount:
                continue
            if amount_filter.max_amount is not None and amount > amount_filter.max_amount:
                continue
            return True
        return False

    def _score_message(
        self,
        *,
        query: str,
        query_terms: list[str],
        combined_text: str,
        fuzzy_matches: list[FuzzyMatch],
        mode: SearchMode,
    ) -> float:
        normalized_text = normalize_search_text(combined_text)
        normalized_query = normalize_search_text(query)
        text_terms = set(tokenize(combined_text))
        keyword_hits = sum(1 for term in query_terms if term in text_terms)
        semantic_score = SequenceMatcher(None, normalized_query, normalized_text).ratio()
        fuzzy_bonus = float(len(fuzzy_matches)) * 0.3

        if mode == SearchMode.KEYWORD:
            return float(keyword_hits) + fuzzy_bonus
        if mode == SearchMode.SEMANTIC:
            return semantic_score * 10.0 + fuzzy_bonus
        return float(keyword_hits) * 1.5 + semantic_score * 8.0 + fuzzy_bonus

    def _make_snippet(self, text: str, query_terms: list[str]) -> str:
        normalized_text = text.replace("\n", " ")
        lowered = normalized_text.casefold()
        for term in query_terms:
            position = lowered.find(term.casefold())
            if position != -1:
                start = max(0, position - 40)
                end = min(len(normalized_text), position + 120)
                return self._apply_privacy(normalized_text[start:end].strip())
        return self._apply_privacy(normalized_text[:160].strip())

    def _build_references(
        self,
        results: Sequence[EmailSearchResult],
        config: MailQueryConfig,
    ) -> list[EmailRef]:
        message_map = {message.message_id: message for message in self._messages}
        references: list[EmailRef] = []
        for result in results[: config.max_references]:
            message = message_map.get(result.message_id)
            if message is None:
                continue
            attachment_ref = self._find_attachment_reference(message, config.query)
            snippet = self._apply_privacy(result.snippet)
            sha_value: str | None = None
            if self._privacy.include_verbatim_hashes:
                sha_value = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
            references.append(
                EmailRef(
                    message_id=message.message_id,
                    thread_id=message.thread_id,
                    connector_id=message.connector_id,
                    mailbox=message.mailbox,
                    subject=message.subject,
                    sender=message.sender.display,
                    sent_at=_ensure_utc(message.sent_at),
                    snippet=snippet,
                    attachment=attachment_ref,
                    sha256=sha_value,
                )
            )
        return references

    def _find_attachment_reference(
        self,
        message: MailMessage,
        query: str,
    ) -> AttachmentRef | None:
        query_terms = tokenize(query)
        for attachment in message.attachments:
            content = attachment.text_content
            if not content:
                continue
            lowered = content.casefold()
            if any(term in lowered for term in query_terms):
                snippet = self._make_snippet(content, query_terms)
                return AttachmentRef(
                    filename=attachment.filename,
                    content_type=attachment.content_type,
                    snippet=self._apply_privacy(snippet),
                )
        return None

    def _build_answer(
        self,
        query: str,
        results: Sequence[EmailSearchResult],
        references: Sequence[EmailRef],
        verbatim: bool,
    ) -> str:
        if not results:
            return f"No matching emails found for '{query}'."

        if verbatim:
            lines = [
                (
                    f"{reference.sent_at.date()} | {reference.sender} | "
                    f"{reference.subject}: {reference.snippet}"
                )
                for reference in references
            ]
            return "\n".join(lines)

        # LLM-enhanced answer when a local or cloud kit is attached
        if self._kit is not None:
            return self._llm_answer(query, references)

        top_reference = references[0]
        connector_count = len({result.connector_id for result in results})
        lines = [
            (
                f"Found {len(results)} matching emails across "
                f"{connector_count} connector(s)."
            ),
            (
                f"Strongest hit: {top_reference.subject} from {top_reference.sender} "
                f"on {top_reference.sent_at.date()}."
            ),
            f"Evidence: {top_reference.snippet}",
        ]
        if len(references) > 1:
            lines.append(f"Additional corroborating emails: {len(references) - 1}.")
        return " ".join(lines)

    def _llm_answer(self, query: str, references: Sequence[EmailRef]) -> str:
        """Generate a natural-language answer via the attached LLM kit."""
        context_parts: list[str] = []
        for ref in references:
            context_parts.append(
                f"[{ref.sent_at.date()}] From: {ref.sender} | Subject: {ref.subject}\n"
                f"{ref.snippet}"
            )
        context = "\n\n".join(context_parts)
        prompt = (
            "You are an email intelligence assistant. "
            "Answer the user's question based only on the email excerpts below. "
            "Be concise and factual. If the answer is not in the excerpts, say so.\n\n"
            f"Email excerpts:\n{context}\n\n"
            f"Question: {query}"
        )
        try:
            # Support LocalLLMKit (has .chat returning .content)
            # and standard kits (have .chat returning LLMResponse with .content)
            response = self._kit.chat(prompt)  # type: ignore[union-attr]
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        except Exception:
            # Graceful fallback — never crash ask() due to LLM errors
            return (
                f"Found {len(references)} matching email(s) for '{query}'. "
                f"Strongest hit: {references[0].subject} — {references[0].snippet}"
            )

    def _build_actions(self, results: Sequence[EmailSearchResult]) -> list[MailActionSuggestion]:
        actions: list[MailActionSuggestion] = []
        for result in results[:3]:
            lowered = result.snippet.casefold()
            if "invoice" in lowered or "payment" in lowered:
                actions.append(
                    MailActionSuggestion(
                        action="review_financial_follow_up",
                        reason="Email mentions invoice or payment language.",
                        message_id=result.message_id,
                        connector_id=result.connector_id,
                    )
                )
            elif "meeting" in lowered or "call" in lowered:
                actions.append(
                    MailActionSuggestion(
                        action="prepare_meeting_brief",
                        reason="Email appears to discuss a meeting or call.",
                        message_id=result.message_id,
                        connector_id=result.connector_id,
                    )
                )
        return actions

    def _apply_privacy(self, text: str) -> str:
        if not self._privacy.redact_pii:
            return text
        redacted = text
        patterns = (
            (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "[PAN]"),
            (
                r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b",
                "[GSTIN]",
            ),
            (r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b", "[AADHAAR]"),
            (r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]"),
        )
        for pattern, replacement in patterns:
            redacted = re.sub(pattern, replacement, redacted)
        return redacted


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
