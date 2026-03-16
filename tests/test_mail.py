"""Tests for the RactoMail foundation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ractogateway.mail import (
    AmountFilter,
    FuzzySearch,
    GmailConnector,
    IMAPConnector,
    MailAddress,
    MailAttachment,
    MailMessage,
    MailPrivacyConfig,
    MockMailConnector,
    PythonOperator,
    RactoDAG,
    RactoMailKit,
)


def _message(
    *,
    message_id: str,
    thread_id: str,
    connector_id: str,
    subject: str,
    sender: str,
    sent_at: datetime,
    body_text: str,
    labels: list[str] | None = None,
    attachments: list[MailAttachment] | None = None,
    is_spam: bool = False,
) -> MailMessage:
    return MailMessage(
        message_id=message_id,
        thread_id=thread_id,
        connector_id=connector_id,
        mailbox="INBOX",
        subject=subject,
        sender=MailAddress(email=sender, name=sender.split("@", maxsplit=1)[0].title()),
        sent_at=sent_at,
        body_text=body_text,
        labels=labels or [],
        attachments=attachments or [],
        is_spam=is_spam,
    )


def _build_mail() -> RactoMailKit:
    gmail_messages = [
        _message(
            message_id="m-1",
            thread_id="t-1",
            connector_id="gmail",
            subject="Invoice dispute INV-2024-447",
            sender="mehta@example.com",
            sent_at=datetime(2024, 2, 20, 10, 0, tzinfo=UTC),
            body_text=(
                "Mehta raised a GST dispute on invoice INV-2024-447 for Rs. 118,200. "
                "Please review the 18 percent calculation."
            ),
            labels=["finance", "complaint"],
            attachments=[
                MailAttachment(
                    attachment_id="a-1",
                    filename="invoice_notes.txt",
                    text_content="Invoice INV-2024-447 revised amount Rs. 112,000.",
                )
            ],
        ),
        _message(
            message_id="m-2",
            thread_id="t-2",
            connector_id="gmail",
            subject="Weekly vendor complaints",
            sender="ops@example.com",
            sent_at=datetime(2024, 2, 22, 9, 0, tzinfo=UTC),
            body_text="Complaint summary: Mehta delivery delay and GST mismatch.",
            labels=["complaint"],
        ),
    ]
    imap_messages = [
        _message(
            message_id="m-3",
            thread_id="t-3",
            connector_id="imap",
            subject="Diwali pricing promise",
            sender="sales@example.com",
            sent_at=datetime(2023, 11, 2, 9, 30, tzinfo=UTC),
            body_text="Vendor promised a seasonal discount before Diwali.",
            labels=["sales"],
        ),
        _message(
            message_id="m-4",
            thread_id="t-4",
            connector_id="imap",
            subject="Meeting follow-up",
            sender="ceo@example.com",
            sent_at=datetime(2024, 2, 24, 11, 0, tzinfo=UTC),
            body_text="Schedule a meeting next week to resolve the invoice issue.",
            labels=["meeting"],
        ),
    ]
    duplicate = gmail_messages[1]
    connector = GmailConnector(messages=[*gmail_messages, duplicate])
    imap = IMAPConnector(messages=imap_messages)
    mail = RactoMailKit(connectors=[connector, imap])
    mail.sync()
    return mail


def test_top_level_module_export_is_available() -> None:
    import ractogateway as rg

    assert rg.mail.RactoMailKit is RactoMailKit


def test_sync_search_and_versions_work_together() -> None:
    mail = _build_mail()

    stats = mail.stats()
    assert stats.indexed_messages == 4
    assert stats.indexed_threads == 4

    results = mail.search(
        "invoice dispute",
        regex_patterns={"invoice": r"INV-[0-9]{4}-[0-9]{3}"},
        boolean_query='"invoice" AND "gst"',
        amount_filter=AmountFilter(min_amount=100_000, max_amount=200_000),
        top_k=5,
    )

    assert results
    assert results[0].message_id == "m-1"
    assert results[0].regex_matches[0].matched_text == "INV-2024-447"
    assert 118200.0 in results[0].matched_amounts


def test_ask_builds_references_actions_and_attachment_citations() -> None:
    mail = _build_mail()

    response = mail.ask("invoice revised amount", top_k=3, max_references=2)

    assert response.total_matches >= 1
    assert response.references
    assert response.references[0].attachment is not None
    assert response.references[0].sha256 is not None
    assert response.actions
    assert "Strongest hit" in response.answer


def test_privacy_redaction_and_verbatim_mode() -> None:
    sensitive = _message(
        message_id="m-5",
        thread_id="t-5",
        connector_id="mock",
        subject="Vendor KYC",
        sender="finance@example.com",
        sent_at=datetime(2024, 3, 1, 8, 0, tzinfo=UTC),
        body_text=(
            "PAN AABCM1234D and email vendor@example.com were shared for onboarding."
        ),
    )
    mail = RactoMailKit(
        connectors=[MockMailConnector(messages=[sensitive])],
        privacy=MailPrivacyConfig(redact_pii=True),
    )
    mail.sync()

    response = mail.ask("vendor onboarding", verbatim=True)

    assert "[PAN]" in response.answer
    assert "[EMAIL]" in response.answer


def test_session_memory_and_saved_searches_narrow_follow_up_queries() -> None:
    mail = _build_mail()
    session = mail.start_session(session_id="s-1", user="ceo@example.com")

    first = session.ask("show me complaints")
    second = session.ask("only the complaints from Mehta")

    assert first.references
    assert second.references
    assert any("Mehta" in reference.sender for reference in second.references)

    saved = mail.save_search("mehta_complaints", "Mehta complaint")
    replay = mail.run_saved_search(saved.name)
    assert replay.references
    assert replay.references[0].subject


def test_indian_calendar_and_fuzzy_search_are_applied() -> None:
    mail = _build_mail()

    date_results = mail.search("emails from the week before Diwali 2023", top_k=5)
    assert any(result.message_id == "m-3" for result in date_results)

    fuzzy_response = mail.ask(
        "invoice from Mehtta",
        fuzzy=FuzzySearch(enabled=True, max_distance=2),
    )
    assert fuzzy_response.fuzzy_matches
    assert any(match.original == "mehtta" for match in fuzzy_response.fuzzy_matches)


def test_dag_executes_dependencies_and_tracks_outputs() -> None:
    dag = RactoDAG(dag_id="mail_query_dag", max_parallel_tasks=2)
    fetch = dag.add_task(
        PythonOperator(task_id="fetch", python_callable=lambda _ctx: ["m-1", "m-2"])
    )
    classify = dag.add_task(
        PythonOperator(task_id="classify", python_callable=lambda _ctx: "finance")
    )
    combine = dag.add_task(
        PythonOperator(
            task_id="combine",
            python_callable=lambda ctx: (
                len(ctx["task_outputs"]["fetch"]),
                ctx["task_outputs"]["classify"],
            ),
        )
    )
    fetch >> combine
    classify >> combine

    run = dag.run()

    assert run.status == "SUCCESS"
    assert run.task_runs["combine"].output == (2, "finance")


@pytest.mark.asyncio
async def test_async_sync_and_aask_work() -> None:
    connector = MockMailConnector(
        messages=[
            _message(
                message_id="m-6",
                thread_id="t-6",
                connector_id="mock",
                subject="Async invoice",
                sender="async@example.com",
                sent_at=datetime(2024, 2, 1, 10, 0, tzinfo=UTC),
                body_text="Async invoice reminder for Rs. 50,000.",
            )
        ]
    )
    mail = RactoMailKit(connectors=[connector])

    status = await mail.async_sync(mode="active")
    response = await mail.aask("async invoice")

    assert status.indexed_count == 1
    assert response.references
    assert response.references[0].message_id == "m-6"


def test_thread_reconstruction_with_sentiment_arc() -> None:
    connector = MockMailConnector(
        messages=[
            _message(
                message_id="th-1",
                thread_id="t-thread",
                connector_id="mock",
                subject="Invoice Discussion",
                sender="alice@example.com",
                sent_at=datetime(2024, 3, 1, 9, 0, tzinfo=UTC),
                body_text=(
                    "Hi, I will send the invoice by end of week. "
                    "Please confirm receipt. Best regards, Alice"
                ),
            ),
            _message(
                message_id="th-2",
                thread_id="t-thread",
                connector_id="mock",
                subject="Re: Invoice Discussion",
                sender="bob@example.com",
                sent_at=datetime(2024, 3, 2, 10, 0, tzinfo=UTC),
                body_text=(
                    "Thank you Alice, we agreed to the revised amount. "
                    "Please proceed with the invoice."
                ),
            ),
        ]
    )
    mail = RactoMailKit(connectors=[connector])
    mail.sync()

    thread = mail.get_thread("t-thread")

    assert thread.thread_id == "t-thread"
    assert len(thread.messages) == 2
    assert thread.messages[0].position == 1
    assert thread.messages[1].position == 2
    assert len(thread.participants) == 2
    assert len(thread.response_times_hours) == 1
    assert thread.response_times_hours[0] > 0
    assert "positive" in thread.sentiment_arc or "neutral" in thread.sentiment_arc
    assert thread.summary


def test_commitment_detection() -> None:
    connector = MockMailConnector(
        messages=[
            _message(
                message_id="c-1",
                thread_id="t-commit",
                connector_id="mock",
                subject="Delivery Promise",
                sender="vendor@example.com",
                sent_at=datetime(2024, 3, 5, 9, 0, tzinfo=UTC),
                body_text=(
                    "I will send the goods by Friday. "
                    "We'll follow up with the tracking number."
                ),
            )
        ]
    )
    mail = RactoMailKit(connectors=[connector])
    mail.sync()

    commitments = mail.get_commitments()

    assert commitments
    assert all(c.message_id == "c-1" for c in commitments)
    assert all(c.confidence > 0 for c in commitments)
    assert all(c.commitment_id for c in commitments)


def test_unanswered_detection() -> None:
    old_message = _message(
        message_id="u-1",
        thread_id="t-unanswered",
        connector_id="mock",
        subject="Pending query",
        sender="client@example.com",
        sent_at=datetime(2023, 1, 1, 9, 0, tzinfo=UTC),
        body_text="Please let me know the status.",
    )
    connector = MockMailConnector(messages=[old_message])
    mail = RactoMailKit(connectors=[connector])
    mail.sync()

    unanswered = mail.get_unanswered(min_age_hours=1.0)

    assert unanswered
    assert unanswered[0].message_id == "u-1"
    assert unanswered[0].age_days > 0


def test_contact_profile_builds_correctly() -> None:
    mail = _build_mail()

    profile = mail.contact_profile("mehta@example.com")

    assert profile.email == "mehta@example.com"
    assert profile.total_emails >= 1
    assert profile.sentiment_trend in ("positive", "neutral", "negative", "mixed")
    assert 0.0 <= profile.relationship_health_score <= 1.0
    assert profile.business_relationship_summary


def test_negotiation_tracking_detects_rounds() -> None:
    connector = MockMailConnector(
        messages=[
            _message(
                message_id="n-1",
                thread_id="t-neg",
                connector_id="mock",
                subject="Price Negotiation",
                sender="buyer@example.com",
                sent_at=datetime(2024, 4, 1, 9, 0, tzinfo=UTC),
                body_text="We'd like to offer Rs. 50,000 for the contract.",
            ),
            _message(
                message_id="n-2",
                thread_id="t-neg",
                connector_id="mock",
                subject="Re: Price Negotiation",
                sender="seller@example.com",
                sent_at=datetime(2024, 4, 2, 10, 0, tzinfo=UTC),
                body_text="Counter proposal: Rs. 65,000 is our final price.",
            ),
        ]
    )
    mail = RactoMailKit(connectors=[connector])
    mail.sync()

    result = mail.get_negotiation("t-neg")

    assert result.thread_id == "t-neg"
    assert len(result.rounds) == 2
    assert result.rounds[0].price_mentioned == 50000.0
    assert result.rounds[1].price_mentioned == 65000.0
    assert result.final_price == 65000.0


def test_sentiment_timeline_builds_arc() -> None:
    connector = MockMailConnector(
        messages=[
            _message(
                message_id="s-1",
                thread_id="t-s1",
                connector_id="mock",
                subject="Happy email",
                sender="contact@example.com",
                sent_at=datetime(2024, 5, 1, 9, 0, tzinfo=UTC),
                body_text="Thank you, this is excellent! We are very satisfied.",
            ),
            _message(
                message_id="s-2",
                thread_id="t-s2",
                connector_id="mock",
                subject="Complaint email",
                sender="contact@example.com",
                sent_at=datetime(2024, 5, 5, 9, 0, tzinfo=UTC),
                body_text="There is a serious problem and we are very disappointed.",
            ),
        ]
    )
    mail = RactoMailKit(connectors=[connector])
    mail.sync()

    timeline = mail.sentiment_timeline("contact@example.com")

    assert timeline.contact_email == "contact@example.com"
    assert len(timeline.points) == 2
    assert timeline.overall in ("positive", "negative", "neutral")
    assert timeline.risk_level in ("low", "medium", "high")


def test_daily_briefing_generates_summary() -> None:
    mail = _build_mail()

    briefing = mail.daily_briefing()

    assert briefing.total_emails == 4
    assert briefing.summary
    assert isinstance(briefing.pending_commitments, list)
    assert isinstance(briefing.unanswered_emails, list)
    assert isinstance(briefing.top_threads, list)


def test_spam_filter_blocks_blacklisted_sender() -> None:
    from ractogateway.mail import SpamConfig

    spam_msg = _message(
        message_id="spam-1",
        thread_id="t-spam",
        connector_id="mock",
        subject="Buy now!",
        sender="spammer@spam.com",
        sent_at=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
        body_text="Click here to unsubscribe from this promotional newsletter.",
    )
    clean_msg = _message(
        message_id="clean-1",
        thread_id="t-clean",
        connector_id="mock",
        subject="Regular email",
        sender="clean@example.com",
        sent_at=datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
        body_text="Here is the quarterly report.",
    )
    connector = MockMailConnector(messages=[spam_msg, clean_msg])
    mail = RactoMailKit(
        connectors=[connector],
        spam_config=SpamConfig(
            sender_blacklist=["spammer@spam.com"],
            block_promotional=True,
        ),
    )
    mail.sync()

    assert mail.stats().indexed_messages == 1
    assert mail.stats().spam_filtered_count == 1


def test_security_audit_log_records_operations() -> None:
    from ractogateway.mail import SecurityConfig

    connector = MockMailConnector(
        messages=[
            _message(
                message_id="sec-1",
                thread_id="t-sec",
                connector_id="mock",
                subject="Secure email",
                sender="user@example.com",
                sent_at=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
                body_text="Internal confidential content.",
            )
        ]
    )
    mail = RactoMailKit(
        connectors=[connector],
        security_config=SecurityConfig(audit_logging=True),
    )
    mail.sync()
    mail.ask("confidential")

    log = mail.audit_log()
    assert log
    assert any(e.operation == "ask" for e in log)
    assert mail.stats().audit_log_entries >= 1


def test_right_to_forget_removes_messages() -> None:
    connector = MockMailConnector(
        messages=[
            _message(
                message_id="rtf-1",
                thread_id="t-rtf",
                connector_id="mock",
                subject="GDPR request",
                sender="delete_me@example.com",
                sent_at=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
                body_text="Please delete my data.",
            ),
            _message(
                message_id="rtf-2",
                thread_id="t-rtf2",
                connector_id="mock",
                subject="Keep this",
                sender="keep@example.com",
                sent_at=datetime(2024, 1, 2, 9, 0, tzinfo=UTC),
                body_text="This email should remain.",
            ),
        ]
    )
    mail = RactoMailKit(connectors=[connector])
    mail.sync()

    assert mail.stats().indexed_messages == 2

    removed = mail.forget("delete_me@example.com")

    assert removed == 1
    assert mail.stats().indexed_messages == 1


def test_analytics_computes_sentiment_and_duplicates() -> None:
    mail = _build_mail()

    result = mail.analytics()

    assert result.total_emails == 4
    assert result.unique_contacts >= 1
    assert "positive" in result.sentiment_distribution or "neutral" in result.sentiment_distribution
    assert isinstance(result.top_senders, list)


def test_html_table_extraction() -> None:
    connector = MockMailConnector(
        messages=[
            _message(
                message_id="html-1",
                thread_id="t-html",
                connector_id="mock",
                subject="Invoice Table",
                sender="billing@example.com",
                sent_at=datetime(2024, 3, 1, 9, 0, tzinfo=UTC),
                body_text=(
                    "<table><tr><th>Item</th><th>Amount</th></tr>"
                    "<tr><td>GST</td><td>18,000</td></tr>"
                    "<tr><td>Base</td><td>100,000</td></tr></table>"
                ),
            )
        ]
    )
    mail = RactoMailKit(connectors=[connector])
    mail.sync()

    tables = mail.extract_html_tables("html-1")

    assert tables
    assert tables[0].headers == ["Item", "Amount"]
    assert len(tables[0].rows) == 2
    assert tables[0].inferred_type == "invoice"
    assert 1 in tables[0].numeric_columns


def test_mail_response_has_new_fields() -> None:
    mail = _build_mail()

    response = mail.ask("invoice")

    assert hasattr(response, "confidence_score")
    assert hasattr(response, "sources_searched")
    assert hasattr(response, "query_time_ms")
    assert hasattr(response, "missing_data_explanation")
    assert 0.0 <= response.confidence_score <= 1.0
    assert response.query_time_ms is not None
    assert response.query_time_ms >= 0


def test_delta_checkpoints_return_per_connector() -> None:
    from ractogateway.mail import GmailConnector, IMAPConnector

    mail = RactoMailKit(connectors=[GmailConnector(), IMAPConnector()])
    checkpoints = mail.delta_checkpoints()

    assert len(checkpoints) == 2
    assert {cp.connector_id for cp in checkpoints} == {"gmail", "imap"}


def test_indian_currency_normalization() -> None:
    from ractogateway.mail import normalize_indian_currency

    result = normalize_indian_currency("Rs. 5 lakh")
    assert "500000" in result

    result2 = normalize_indian_currency("₹2 crore")
    assert "20000000" in result2


def test_language_mixing_detection() -> None:
    from ractogateway.mail import detect_language_mixing

    mixed = "Please send the invoice रुपए 50,000"
    latin_only = "Please send the invoice Rs. 50,000"

    assert detect_language_mixing(mixed) is True
    assert detect_language_mixing(latin_only) is False


def test_mailbox_chain_model() -> None:
    from ractogateway.mail import MailboxChain

    chain = MailboxChain(
        name="support_chain",
        connector_ids=["support_gmail", "support_imap"],
        description="All support mailboxes",
    )
    assert chain.name == "support_chain"
    assert len(chain.connector_ids) == 2


def test_commitment_and_unanswered_in_daily_briefing() -> None:
    old = _message(
        message_id="db-1",
        thread_id="t-db1",
        connector_id="mock",
        subject="Old unanswered",
        sender="client@example.com",
        sent_at=datetime(2023, 6, 1, 9, 0, tzinfo=UTC),
        body_text="We'll deliver the report by Friday.",
    )
    connector = MockMailConnector(messages=[old])
    mail = RactoMailKit(connectors=[connector])
    mail.sync()

    briefing = mail.daily_briefing()

    assert briefing.total_emails == 1
    assert briefing.pending_commitments or briefing.unanswered_emails  # at least one detected
    assert briefing.summary
