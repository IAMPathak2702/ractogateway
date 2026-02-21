"""Pydantic models for Kafka integration primitives."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class KafkaProducerConfig(BaseModel):
    """Configuration for :class:`KafkaProducerClient`.

    Parameters
    ----------
    bootstrap_servers:
        Kafka broker list (single host string or multiple hosts).
    client_id:
        Producer client ID reported to the broker.
    acks:
        Acknowledgement policy; ``"all"`` is safest for durability.
    linger_ms:
        Small batching delay to improve throughput.
    compression_type:
        Optional compression algorithm (e.g. ``"gzip"``, ``"snappy"``).
    retries:
        Retry attempts for transient broker/network failures.
    request_timeout_ms:
        Broker request timeout.
    security_protocol:
        Kafka transport protocol (``PLAINTEXT``, ``SSL``, ``SASL_PLAINTEXT``,
        ``SASL_SSL``).
    sasl_mechanism:
        SASL mechanism when using SASL security protocols.
    sasl_plain_username:
        Username for SASL PLAIN/SCRAM.
    sasl_plain_password:
        Password for SASL PLAIN/SCRAM.
    extra:
        Additional kwargs forwarded directly to ``kafka.KafkaProducer``.
    """

    bootstrap_servers: str | list[str] = Field(default="localhost:9092")
    client_id: str = Field(default="ractogateway-producer")
    acks: Literal["all", "0", "1"] | int = Field(default="all")
    linger_ms: int = Field(default=5, ge=0)
    compression_type: str | None = Field(default="gzip")
    retries: int = Field(default=3, ge=0)
    request_timeout_ms: int = Field(default=30_000, gt=0)
    security_protocol: str = Field(default="PLAINTEXT")
    sasl_mechanism: str | None = Field(default=None)
    sasl_plain_username: str | None = Field(default=None)
    sasl_plain_password: str | None = Field(default=None)
    extra: dict[str, Any] = Field(default_factory=dict)


class KafkaConsumerConfig(BaseModel):
    """Configuration for :class:`KafkaConsumerClient`.

    Parameters
    ----------
    topics:
        Topic list to subscribe on startup.
    bootstrap_servers:
        Kafka broker list.
    group_id:
        Consumer group ID.  ``None`` disables group coordination.
    auto_offset_reset:
        Initial offset policy when no committed offset is present.
    enable_auto_commit:
        Whether Kafka should auto-commit offsets in the background.
    max_poll_records:
        Upper bound on records returned per poll call.
    poll_timeout_ms:
        Default poll timeout in milliseconds.
    session_timeout_ms:
        Group session timeout.
    client_id:
        Consumer client ID.
    security_protocol:
        Kafka transport protocol.
    sasl_mechanism:
        SASL mechanism when applicable.
    sasl_plain_username:
        Username for SASL auth.
    sasl_plain_password:
        Password for SASL auth.
    extra:
        Additional kwargs forwarded to ``kafka.KafkaConsumer``.
    """

    topics: list[str] = Field(min_length=1)
    bootstrap_servers: str | list[str] = Field(default="localhost:9092")
    group_id: str | None = Field(default=None)
    auto_offset_reset: Literal["earliest", "latest", "none"] = Field(default="latest")
    enable_auto_commit: bool = Field(default=False)
    max_poll_records: int = Field(default=500, gt=0)
    poll_timeout_ms: int = Field(default=1_000, ge=0)
    session_timeout_ms: int = Field(default=10_000, gt=0)
    client_id: str = Field(default="ractogateway-consumer")
    security_protocol: str = Field(default="PLAINTEXT")
    sasl_mechanism: str | None = Field(default=None)
    sasl_plain_username: str | None = Field(default=None)
    sasl_plain_password: str | None = Field(default=None)
    extra: dict[str, Any] = Field(default_factory=dict)


class KafkaStreamConfig(BaseModel):
    """Batch processing configuration for :class:`KafkaStreamProcessor`.

    Parameters
    ----------
    batch_size:
        Maximum messages to process per handler invocation.
    max_wait_ms:
        Max wall-clock wait to collect one batch.
    poll_timeout_ms:
        Per-poll timeout while collecting a batch.
    commit_on_success:
        Commit offsets after successful handler + publish cycle.
    wait_for_publish:
        If ``True``, block for broker acknowledgement on each produced output.
    """

    batch_size: int = Field(default=32, gt=0)
    max_wait_ms: int = Field(default=250, ge=0)
    poll_timeout_ms: int = Field(default=100, ge=0)
    commit_on_success: bool = Field(default=True)
    wait_for_publish: bool = Field(default=True)


class KafkaMessage(BaseModel):
    """Normalized Kafka record returned by :class:`KafkaConsumerClient`."""

    topic: str
    partition: int
    offset: int
    timestamp_ms: int | None = None
    key: str | bytes | None = None
    value: Any = None
    headers: dict[str, bytes | None] = Field(default_factory=dict)


class KafkaProduceResult(BaseModel):
    """Broker acknowledgement metadata returned by publish operations."""

    topic: str
    partition: int
    offset: int
    timestamp_ms: int | None = None


class KafkaPublishRequest(BaseModel):
    """Explicit publish request used by stream routing handlers."""

    topic: str
    value: Any
    key: str | bytes | None = None
    headers: dict[str, str | bytes | None] | None = None
    partition: int | None = None
    timestamp_ms: int | None = None


class KafkaAuditEvent(BaseModel):
    """Immutable audit envelope for prompt/response traces."""

    user_id: str
    model: str
    prompt: str
    response: str
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
