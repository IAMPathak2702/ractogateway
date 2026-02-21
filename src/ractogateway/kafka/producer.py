"""Kafka producer wrapper for high-throughput event publishing."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from ractogateway.kafka._models import KafkaProducerConfig, KafkaProduceResult

Serializer = Callable[[Any], bytes]


def _require_kafka() -> Any:
    try:
        import kafka as kafka_lib
    except ImportError as exc:
        raise ImportError(
            "The 'kafka-python' package is required for Kafka support. "
            "Install it with:  pip install ractogateway[kafka]"
        ) from exc
    return kafka_lib


def _default_value_serializer(value: Any) -> bytes:
    """Serialize values to UTF-8 bytes for Kafka transport."""
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _default_key_serializer(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _normalise_headers(
    headers: dict[str, str | bytes | None] | None,
) -> list[tuple[str, bytes | None]] | None:
    if headers is None:
        return None
    normalized: list[tuple[str, bytes | None]] = []
    for name, value in headers.items():
        if value is None:
            normalized.append((name, None))
        elif isinstance(value, bytes):
            normalized.append((name, value))
        else:
            normalized.append((name, value.encode("utf-8")))
    return normalized


class KafkaProducerClient:
    """Typed facade over ``kafka.KafkaProducer``.

    Parameters
    ----------
    config:
        Producer connection and reliability settings. Defaults are applied when
        omitted.
    producer:
        Pre-built Kafka producer object. Useful for dependency injection in
        tests or when your runtime provides the producer lifecycle externally.
    key_serializer:
        Optional override for key serialization.
    value_serializer:
        Optional override for value serialization.
    """

    def __init__(
        self,
        *,
        config: KafkaProducerConfig | None = None,
        producer: Any | None = None,
        key_serializer: Serializer | None = None,
        value_serializer: Serializer | None = None,
    ) -> None:
        self._config = config or KafkaProducerConfig()
        self._provided_producer = producer
        self._producer: Any | None = None
        self._key_serializer = key_serializer or _default_key_serializer
        self._value_serializer = value_serializer or _default_value_serializer

    def _client(self) -> Any:
        if self._provided_producer is not None:
            return self._provided_producer
        if self._producer is not None:
            return self._producer

        kafka_lib = _require_kafka()
        kwargs: dict[str, Any] = {
            "bootstrap_servers": self._config.bootstrap_servers,
            "client_id": self._config.client_id,
            "acks": self._config.acks,
            "linger_ms": self._config.linger_ms,
            "compression_type": self._config.compression_type,
            "retries": self._config.retries,
            "request_timeout_ms": self._config.request_timeout_ms,
            "security_protocol": self._config.security_protocol,
            "key_serializer": self._key_serializer,
            "value_serializer": self._value_serializer,
        }
        if self._config.sasl_mechanism is not None:
            kwargs["sasl_mechanism"] = self._config.sasl_mechanism
        if self._config.sasl_plain_username is not None:
            kwargs["sasl_plain_username"] = self._config.sasl_plain_username
        if self._config.sasl_plain_password is not None:
            kwargs["sasl_plain_password"] = self._config.sasl_plain_password
        kwargs.update(self._config.extra)

        self._producer = kafka_lib.KafkaProducer(**kwargs)
        return self._producer

    def publish(
        self,
        topic: str,
        value: Any,
        *,
        key: str | bytes | None = None,
        headers: dict[str, str | bytes | None] | None = None,
        partition: int | None = None,
        timestamp_ms: int | None = None,
        wait: bool = True,
        timeout_s: float = 10.0,
    ) -> KafkaProduceResult | None:
        """Publish one event and optionally wait for broker acknowledgement."""
        future = self._client().send(
            topic,
            value=value,
            key=key,
            headers=_normalise_headers(headers),
            partition=partition,
            timestamp_ms=timestamp_ms,
        )
        if not wait:
            return None
        metadata = future.get(timeout=timeout_s)
        return KafkaProduceResult(
            topic=cast("str", metadata.topic),
            partition=cast("int", metadata.partition),
            offset=cast("int", metadata.offset),
            timestamp_ms=cast("int | None", getattr(metadata, "timestamp", None)),
        )

    def publish_json(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        key: str | bytes | None = None,
        headers: dict[str, str | bytes | None] | None = None,
        partition: int | None = None,
        timestamp_ms: int | None = None,
        wait: bool = True,
        timeout_s: float = 10.0,
    ) -> KafkaProduceResult | None:
        """Alias for :meth:`publish` when value is a JSON payload."""
        return self.publish(
            topic,
            payload,
            key=key,
            headers=headers,
            partition=partition,
            timestamp_ms=timestamp_ms,
            wait=wait,
            timeout_s=timeout_s,
        )

    def flush(self, *, timeout_s: float | None = None) -> None:
        """Force all buffered events to be sent to brokers."""
        if timeout_s is None:
            self._client().flush()
            return
        self._client().flush(timeout=timeout_s)

    def close(self, *, timeout_s: float = 5.0) -> None:
        """Close underlying producer and release network resources."""
        self._client().close(timeout=timeout_s)

    def __enter__(self) -> KafkaProducerClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

