"""Kafka consumer wrapper that returns normalized typed records."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ractogateway.kafka._models import KafkaConsumerConfig, KafkaMessage

Deserializer = Callable[[Any], Any]
HEADER_TUPLE_SIZE = 2


def _require_kafka() -> Any:
    try:
        import kafka as kafka_lib
    except ImportError as exc:
        raise ImportError(
            "The 'kafka-python' package is required for Kafka support. "
            "Install it with:  pip install ractogateway[kafka]"
        ) from exc
    return kafka_lib


def _default_key_deserializer(value: Any) -> str | bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if isinstance(value, str):
        return value
    return str(value)


def _default_value_deserializer(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    elif isinstance(value, str):
        text = value
    else:
        return value

    stripped = text.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return text


def _normalize_headers(raw_headers: Any) -> dict[str, bytes | None]:
    if raw_headers is None:
        return {}
    out: dict[str, bytes | None] = {}
    for item in raw_headers:
        if not isinstance(item, tuple) or len(item) != HEADER_TUPLE_SIZE:
            continue
        name, value = item
        if not isinstance(name, str):
            continue
        if value is None or isinstance(value, bytes):
            out[name] = value
    return out


class KafkaConsumerClient:
    """Typed facade over ``kafka.KafkaConsumer``.

    Parameters
    ----------
    config:
        Consumer connection and polling options.
    consumer:
        Pre-built consumer object; useful for tests and dependency injection.
    key_deserializer:
        Optional override for key decoding.
    value_deserializer:
        Optional override for value decoding.
    """

    def __init__(
        self,
        *,
        config: KafkaConsumerConfig,
        consumer: Any | None = None,
        key_deserializer: Deserializer | None = None,
        value_deserializer: Deserializer | None = None,
    ) -> None:
        self._config = config
        self._provided_consumer = consumer
        self._consumer: Any | None = None
        self._key_deserializer = key_deserializer or _default_key_deserializer
        self._value_deserializer = value_deserializer or _default_value_deserializer

    def _client(self) -> Any:
        if self._provided_consumer is not None:
            return self._provided_consumer
        if self._consumer is not None:
            return self._consumer

        kafka_lib = _require_kafka()
        kwargs: dict[str, Any] = {
            "bootstrap_servers": self._config.bootstrap_servers,
            "group_id": self._config.group_id,
            "auto_offset_reset": self._config.auto_offset_reset,
            "enable_auto_commit": self._config.enable_auto_commit,
            "max_poll_records": self._config.max_poll_records,
            "session_timeout_ms": self._config.session_timeout_ms,
            "client_id": self._config.client_id,
            "security_protocol": self._config.security_protocol,
            "key_deserializer": self._key_deserializer,
            "value_deserializer": self._value_deserializer,
        }
        if self._config.sasl_mechanism is not None:
            kwargs["sasl_mechanism"] = self._config.sasl_mechanism
        if self._config.sasl_plain_username is not None:
            kwargs["sasl_plain_username"] = self._config.sasl_plain_username
        if self._config.sasl_plain_password is not None:
            kwargs["sasl_plain_password"] = self._config.sasl_plain_password
        kwargs.update(self._config.extra)

        self._consumer = kafka_lib.KafkaConsumer(*self._config.topics, **kwargs)
        return self._consumer

    def poll(
        self,
        *,
        timeout_ms: int | None = None,
        max_records: int | None = None,
    ) -> list[KafkaMessage]:
        """Poll records and return normalized :class:`KafkaMessage` entries."""
        timeout = self._config.poll_timeout_ms if timeout_ms is None else timeout_ms
        max_rows = self._config.max_poll_records if max_records is None else max_records

        raw_batches = self._client().poll(timeout_ms=timeout, max_records=max_rows)
        out: list[KafkaMessage] = []
        for records in raw_batches.values():
            for record in records:
                out.append(
                    KafkaMessage(
                        topic=record.topic,
                        partition=record.partition,
                        offset=record.offset,
                        timestamp_ms=getattr(record, "timestamp", None),
                        key=record.key,
                        value=record.value,
                        headers=_normalize_headers(getattr(record, "headers", None)),
                    )
                )
        return out

    def commit(self) -> None:
        """Commit currently consumed offsets."""
        self._client().commit()

    def close(self) -> None:
        """Close underlying consumer and release resources."""
        self._client().close()

    def __enter__(self) -> KafkaConsumerClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
