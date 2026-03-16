"""Typed models for the RactoMail platform."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    """Search strategy used by the query engine."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SyncMode(str, Enum):
    """Synchronization strategies exposed by the kit."""

    PASSIVE = "passive"
    ACTIVE = "active"
    DELTA = "delta"
    FULL = "full"


class MailAddress(BaseModel):
    """Structured email address."""

    email: str
    name: str | None = None

    @property
    def display(self) -> str:
        """Return a readable sender/recipient string."""
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email


class AttachmentType(str, Enum):
    """Attachment file type classification."""

    PDF = "pdf"
    WORD = "word"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    HTML = "html"
    IMAGE = "image"
    AUDIO = "audio"
    ZIP = "zip"
    RAR = "rar"
    RTF = "rtf"
    TEXT = "text"
    EML = "eml"
    MSG = "msg"
    MARKDOWN = "markdown"
    ODT = "odt"
    ODS = "ods"
    ODP = "odp"
    UNKNOWN = "unknown"


class HtmlTableData(BaseModel):
    """Extracted HTML table structure."""

    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    numeric_columns: list[int] = Field(default_factory=list)
    inferred_type: str = "generic"  # invoice, bank_statement, generic


class MailAttachment(BaseModel):
    """Normalized attachment content."""

    attachment_id: str
    filename: str
    content_type: str = "text/plain"
    text_content: str = ""
    size_bytes: int = 0
    # Extended fields
    attachment_type: AttachmentType = AttachmentType.UNKNOWN
    page_count: int | None = None
    sheets: list[str] = Field(default_factory=list)
    row_count: int | None = None
    tables: list[HtmlTableData] = Field(default_factory=list)
    ocr_text: str | None = None
    transcript: str | None = None  # audio transcription
    language: str | None = None
    named_ranges: list[str] = Field(default_factory=list)


class MailMessage(BaseModel):
    """Normalized email record emitted by any connector."""

    message_id: str
    thread_id: str
    connector_id: str
    mailbox: str = "INBOX"
    subject: str
    sender: MailAddress
    recipients: list[MailAddress] = Field(default_factory=list)
    cc: list[MailAddress] = Field(default_factory=list)
    bcc: list[MailAddress] = Field(default_factory=list)
    sent_at: datetime
    body_text: str
    labels: list[str] = Field(default_factory=list)
    attachments: list[MailAttachment] = Field(default_factory=list)
    is_spam: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)

    def combined_text(self) -> str:
        """Return the body plus attachment text."""
        attachment_text = "\n".join(
            attachment.text_content for attachment in self.attachments if attachment.text_content
        )
        return "\n".join(
            part
            for part in (self.subject, self.body_text, attachment_text)
            if part
        )


class MailPrivacyConfig(BaseModel):
    """Privacy and output controls."""

    redact_pii: bool = False
    include_verbatim_hashes: bool = True
    strict_read_only: bool = True


class DateRange(BaseModel):
    """Resolved date range from a natural-language calendar reference."""

    label: str
    start: datetime
    end: datetime


class FuzzySearch(BaseModel):
    """Fuzzy search options."""

    enabled: bool = True
    max_distance: int = 2
    apply_to: list[str] = Field(default_factory=lambda: ["subject", "body", "attachments"])
    indian_names: bool = True
    transliteration_fuzzy: bool = True


class AmountFilter(BaseModel):
    """Filter search results by extracted amount values."""

    min_amount: float | None = None
    max_amount: float | None = None
    currency: str = "INR"


class MailFacets(BaseModel):
    """Search facets reused across ask/search/session APIs."""

    connectors: list[str] | None = None
    senders: list[str] | None = None
    labels: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    include_spam: bool = False
    message_ids: list[str] | None = None


class MailQueryConfig(BaseModel):
    """Canonical query model for the mail engine."""

    query: str
    search_mode: SearchMode = SearchMode.HYBRID
    top_k: int = 5
    max_references: int = 5
    verbatim: bool = False
    fuzzy: FuzzySearch | None = None
    regex_patterns: dict[str, str] = Field(default_factory=dict)
    regex_match_mode: str = "any"
    boolean_query: str | None = None
    amount_filter: AmountFilter | None = None
    facets: MailFacets = Field(default_factory=MailFacets)


class AttachmentRef(BaseModel):
    """Reference into a specific attachment."""

    filename: str
    content_type: str
    snippet: str
    # Extended fields
    sheet: str | None = None
    row_range: str | None = None
    cell_range: str | None = None
    page_number: int | None = None
    ocr_text: str | None = None


class EmailRef(BaseModel):
    """Surgical email citation used in answers."""

    message_id: str
    thread_id: str
    connector_id: str
    mailbox: str
    subject: str
    sender: str
    sent_at: datetime
    snippet: str
    attachment: AttachmentRef | None = None
    sha256: str | None = None
    # Extended fields
    confidence_score: float = 0.0
    thread_position: int | None = None


class EvidenceItem(BaseModel):
    """Small trace that explains how the answer was grounded."""

    summary: str
    reference: EmailRef


class RegexMatch(BaseModel):
    """Concrete regex hit inside a message."""

    pattern_name: str
    matched_text: str
    start: int
    end: int
    location: str = "body"


class FuzzyMatch(BaseModel):
    """Concrete fuzzy hit for typo-tolerant search."""

    original: str
    matched_to: str
    distance: int
    match_count: int = 1


class EmailSearchResult(BaseModel):
    """Ranked search hit returned by ``search`` and embedded into ``ask``."""

    message_id: str
    thread_id: str
    connector_id: str
    mailbox: str
    subject: str
    sender: str
    sent_at: datetime
    score: float
    snippet: str
    regex_matches: list[RegexMatch] = Field(default_factory=list)
    matched_amounts: list[float] = Field(default_factory=list)


class MailActionSuggestion(BaseModel):
    """V4-style action suggestion derived from email signals."""

    action: str
    reason: str
    message_id: str
    connector_id: str


class FacetedResult(BaseModel):
    """Faceted search response."""

    query: str
    total_matches: int
    connector_counts: dict[str, int] = Field(default_factory=dict)
    sender_counts: dict[str, int] = Field(default_factory=dict)
    label_counts: dict[str, int] = Field(default_factory=dict)
    emails: list[EmailSearchResult] = Field(default_factory=list)


class EntityResult(BaseModel):
    """Entity-centric search response."""

    entity: str
    total_matches: int
    emails: list[EmailSearchResult] = Field(default_factory=list)


class MailResponse(BaseModel):
    """Primary answer object returned by ``ask`` and ``aask``."""

    query: str
    answer: str
    references: list[EmailRef] = Field(default_factory=list)
    evidence_chain: list[EvidenceItem] = Field(default_factory=list)
    search_results: list[EmailSearchResult] = Field(default_factory=list)
    fuzzy_matches: list[FuzzyMatch] = Field(default_factory=list)
    actions: list[MailActionSuggestion] = Field(default_factory=list)
    total_matches: int = 0
    search_mode: SearchMode = SearchMode.HYBRID
    resolved_date_range: DateRange | None = None
    # Extended fields
    confidence_score: float = 0.0
    sources_searched: int = 0
    cache_hit_type: str | None = None
    token_usage: int | None = None
    query_time_ms: float | None = None
    missing_data_explanation: str | None = None
    extracted_structured_data: list[dict[str, Any]] = Field(default_factory=list)


class SavedSearch(BaseModel):
    """Persisted query that can be rerun later."""

    name: str
    config: MailQueryConfig
    created_at: datetime


class ConversationTurn(BaseModel):
    """Persisted turn inside a conversation session."""

    turn_number: int
    query_original: str
    query_resolved: str
    filters_applied: MailFacets
    answer: str
    references: list[EmailRef] = Field(default_factory=list)


class SyncStatus(BaseModel):
    """Synchronization result."""

    mode: SyncMode
    indexed_count: int
    thread_count: int
    connector_counts: dict[str, int] = Field(default_factory=dict)
    deduplicated_count: int = 0
    used_flow: bool = False
    dag_run_id: str | None = None


class MailStats(BaseModel):
    """Operational stats exposed by the kit."""

    indexed_messages: int = 0
    indexed_threads: int = 0
    saved_searches: int = 0
    sessions: int = 0
    # Extended fields
    spam_filtered_count: int = 0
    total_commitments: int = 0
    total_unanswered: int = 0
    audit_log_entries: int = 0


# ── New intelligence models ──────────────────────────────────────────────────


class CommitmentStatus(str, Enum):
    """Status of a detected commitment."""

    PENDING = "pending"
    FULFILLED = "fulfilled"
    OVERDUE = "overdue"
    UNKNOWN = "unknown"


class Commitment(BaseModel):
    """A detected commitment extracted from email text."""

    commitment_id: str
    message_id: str
    thread_id: str
    connector_id: str
    sender: str
    text: str
    due_date: datetime | None = None
    status: CommitmentStatus = CommitmentStatus.UNKNOWN
    confidence: float = 0.0


class ThreadEmail(BaseModel):
    """A single message within a reconstructed thread."""

    message_id: str
    subject: str
    sender: MailAddress
    sent_at: datetime
    body_text: str
    position: int


class EmailThread(BaseModel):
    """Reconstructed email thread with intelligence signals."""

    thread_id: str
    subject: str
    connector_id: str
    participants: list[MailAddress] = Field(default_factory=list)
    messages: list[ThreadEmail] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    open_items: list[str] = Field(default_factory=list)
    response_times_hours: list[float] = Field(default_factory=list)
    sentiment_arc: list[str] = Field(default_factory=list)
    summary: str = ""


class UnansweredEmail(BaseModel):
    """An email that appears to be awaiting a reply."""

    message_id: str
    thread_id: str
    connector_id: str
    subject: str
    sender: str
    sent_at: datetime
    age_days: float
    priority: str = "normal"  # high/normal/low


class CommunicationPattern(BaseModel):
    """Derived communication statistics for a contact."""

    avg_response_hours: float | None = None
    email_frequency_per_month: float | None = None
    most_active_hour: int | None = None


class ContactProfile(BaseModel):
    """Relationship intelligence profile for a single contact."""

    email: str
    name: str | None = None
    total_emails: int = 0
    communication_pattern: CommunicationPattern = Field(default_factory=CommunicationPattern)
    sentiment_trend: str = "neutral"
    business_relationship_summary: str = ""
    unresolved_issues: list[str] = Field(default_factory=list)
    outstanding_commitments: list[Commitment] = Field(default_factory=list)
    response_time_hours: float | None = None
    relationship_health_score: float = 0.5


class NegotiationRound(BaseModel):
    """A single round in a price/term negotiation thread."""

    round_number: int
    message_id: str
    sender: str
    sent_at: datetime
    price_mentioned: float | None = None
    position: str = ""


class NegotiationResult(BaseModel):
    """Analysis of negotiation patterns across a thread."""

    thread_id: str
    subject: str
    rounds: list[NegotiationRound] = Field(default_factory=list)
    final_outcome: str = ""
    final_price: float | None = None
    pattern: str = ""


class SentimentPoint(BaseModel):
    """A single sentiment measurement for a message."""

    message_id: str
    sender: str
    sent_at: datetime
    sentiment: str  # positive/neutral/negative
    score: float  # -1 to 1


class SentimentTimeline(BaseModel):
    """Sentiment timeline for a specific contact."""

    contact_email: str
    points: list[SentimentPoint] = Field(default_factory=list)
    overall: str = "neutral"
    risk_level: str = "low"  # low/medium/high


class DailyBriefing(BaseModel):
    """Daily intelligence briefing generated by the kit."""

    generated_at: datetime
    total_emails: int = 0
    pending_commitments: list[Commitment] = Field(default_factory=list)
    unanswered_emails: list[UnansweredEmail] = Field(default_factory=list)
    top_threads: list[EmailThread] = Field(default_factory=list)
    summary: str = ""


class SpamConfig(BaseModel):
    """Configuration for spam and noise filtering."""

    use_provider_flags: bool = True
    keyword_blacklist: list[str] = Field(default_factory=list)
    sender_whitelist: list[str] = Field(default_factory=list)
    sender_blacklist: list[str] = Field(default_factory=list)
    block_promotional: bool = False
    block_automated: bool = False
    llm_classify: bool = False


class SecurityConfig(BaseModel):
    """Security and privacy configuration."""

    local_only: bool = False
    encrypt_index: bool = False
    audit_logging: bool = False
    excluded_senders: list[str] = Field(default_factory=list)
    excluded_subjects: list[str] = Field(default_factory=list)
    excluded_folders: list[str] = Field(default_factory=list)
    right_to_forget_emails: list[str] = Field(default_factory=list)


class AuditLogEntry(BaseModel):
    """Single entry in the operational audit log."""

    timestamp: datetime
    operation: str
    query: str | None = None
    user: str | None = None
    result_count: int | None = None


class LanguageConfig(BaseModel):
    """Multi-language processing configuration."""

    languages: list[str] = Field(default_factory=lambda: ["en"])
    detect_mixing: bool = True
    normalize_numbers: bool = True
    normalize_currency: bool = True
    normalize_dates: bool = True


class WebhookSyncConfig(BaseModel):
    """Configuration for webhook-based push sync."""

    provider: str  # gmail, outlook
    endpoint_url: str | None = None
    topic_name: str | None = None   # Gmail Pub/Sub
    subscription_id: str | None = None  # Outlook Graph


class DeltaCheckpoint(BaseModel):
    """Delta sync checkpoint for incremental fetching."""

    connector_id: str
    delta_token: str
    last_synced_at: datetime


class EmailAnalytics(BaseModel):
    """Analytics summary computed across indexed messages."""

    period_start: datetime
    period_end: datetime
    total_emails: int = 0
    unique_contacts: int = 0
    avg_response_hours: float | None = None
    sentiment_distribution: dict[str, int] = Field(default_factory=dict)
    top_senders: list[tuple[str, int]] = Field(default_factory=list)
    duplicate_count: int = 0
    relationship_health: dict[str, float] = Field(default_factory=dict)


class MailboxChain(BaseModel):
    """Multi-mailbox chain for support@, sales@, info@ etc."""

    name: str
    connector_ids: list[str]
    description: str = ""
