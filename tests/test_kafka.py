from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ractogateway.kafka import (
    KafkaAuditLogger,
    KafkaConsumerClient,
    KafkaConsumerConfig,
    KafkaProducerClient,
    KafkaProducerConfig,
    KafkaStreamConfig,
    KafkaStreamProcessor,
)


@dataclass
class _FakeMetadata:
    topic: str
    partition: int
    offset: int
    timestamp: int


class _FakeFuture:
    def __init__(self, metadata: _FakeMetadata) -> None:
        self._metadata = metadata

    def get(self, timeout: float | None = None) -> _FakeMetadata:
        _ = timeout
        return self._metadata


class _FakeKafkaProducer:
    last_instance: _FakeKafkaProducer | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.key_serializer = kwargs.get("key_serializer")
        self.value_serializer = kwargs.get("value_serializer")
        self.sent: list[dict[str, Any]] = []
        _FakeKafkaProducer.last_instance = self

    def send(
        self,
        topic: str,
        *,
        value: Any,
        key: Any = None,
        headers: list[tuple[str, bytes | None]] | None = None,
        partition: int | None = None,
        timestamp_ms: int | None = None,
    ) -> _FakeFuture:
        encoded_value = self.value_serializer(value) if self.value_serializer else value
        encoded_key = self.key_serializer(key) if self.key_serializer else key
        self.sent.append(
            {
                "topic": topic,
                "value": encoded_value,
                "key": encoded_key,
                "headers": headers,
                "partition": partition,
                "timestamp_ms": timestamp_ms,
            }
        )
        return _FakeFuture(
            _FakeMetadata(
                topic=topic,
                partition=0 if partition is None else partition,
                offset=len(self.sent) - 1,
                timestamp=timestamp_ms if timestamp_ms is not None else 1_700_000_000_000,
            )
        )

    def flush(self, timeout: float | None = None) -> None:
        _ = timeout

    def close(self, timeout: float = 5.0) -> None:
        _ = timeout


@dataclass
class _FakeConsumerRecord:
    topic: str
    partition: int
    offset: int
    timestamp: int
    key: str | bytes | None
    value: Any
    headers: list[tuple[str, bytes | None]]


class _FakeKafkaConsumer:
    def __init__(self, batches: list[dict[str, list[_FakeConsumerRecord]]]) -> None:
        self._batches = batches
        self.poll_calls: list[tuple[int, int]] = []
        self.commits = 0

    def poll(self, *, timeout_ms: int, max_records: int) -> dict[str, list[_FakeConsumerRecord]]:
        self.poll_calls.append((timeout_ms, max_records))
        if not self._batches:
            return {}
        return self._batches.pop(0)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        return


def test_kafka_producer_client_serializes_payload_and_returns_metadata(monkeypatch: Any) -> None:
    from ractogateway.kafka import producer as producer_module

    fake_module = SimpleNamespace(KafkaProducer=_FakeKafkaProducer)
    monkeypatch.setattr(producer_module, "_require_kafka", lambda: fake_module)

    client = KafkaProducerClient(
        config=KafkaProducerConfig(
            bootstrap_servers="localhost:9092",
            client_id="test-producer",
        )
    )
    result = client.publish(
        "frames.in",
        {"frame_id": 1, "camera_id": "cam-a"},
        key="cam-a",
        headers={"trace_id": "abc123"},
        partition=2,
    )

    assert result is not None
    assert result.topic == "frames.in"
    assert result.partition == 2
    assert result.offset == 0

    sent = _FakeKafkaProducer.last_instance
    assert sent is not None
    assert sent.kwargs["client_id"] == "test-producer"
    assert sent.sent[0]["value"] == b'{"frame_id":1,"camera_id":"cam-a"}'
    assert sent.sent[0]["key"] == b"cam-a"
    assert sent.sent[0]["headers"] == [("trace_id", b"abc123")]


def test_kafka_consumer_client_maps_polled_records() -> None:
    record = _FakeConsumerRecord(
        topic="agent.events",
        partition=1,
        offset=42,
        timestamp=1_700_000_000_100,
        key="research-agent",
        value={"job": "write-summary"},
        headers=[("trace_id", b"req-1")],
    )
    fake_consumer = _FakeKafkaConsumer(batches=[{"tp": [record]}])
    client = KafkaConsumerClient(
        config=KafkaConsumerConfig(topics=["agent.events"]),
        consumer=fake_consumer,
    )

    messages = client.poll(timeout_ms=200, max_records=10)

    assert len(messages) == 1
    assert messages[0].topic == "agent.events"
    assert messages[0].partition == 1
    assert messages[0].offset == 42
    assert messages[0].key == "research-agent"
    assert messages[0].value == {"job": "write-summary"}
    assert messages[0].headers == {"trace_id": b"req-1"}
    assert fake_consumer.poll_calls == [(200, 10)]


def test_kafka_stream_processor_publishes_handler_outputs_and_commits() -> None:
    batch = {
        "tp": [
            _FakeConsumerRecord(
                topic="frames.in",
                partition=0,
                offset=10,
                timestamp=1_700_000_000_010,
                key="cam-1",
                value={"frame_id": 10},
                headers=[],
            ),
            _FakeConsumerRecord(
                topic="frames.in",
                partition=0,
                offset=11,
                timestamp=1_700_000_000_011,
                key="cam-2",
                value={"frame_id": 11},
                headers=[],
            ),
        ]
    }
    fake_consumer = _FakeKafkaConsumer(batches=[batch])
    consumer = KafkaConsumerClient(
        config=KafkaConsumerConfig(topics=["frames.in"]),
        consumer=fake_consumer,
    )
    fake_producer = _FakeKafkaProducer()
    producer = KafkaProducerClient(producer=fake_producer)
    processor = KafkaStreamProcessor(
        consumer=consumer,
        producer=producer,
        config=KafkaStreamConfig(
            batch_size=2,
            max_wait_ms=0,
            wait_for_publish=False,
        ),
    )

    consumed = processor.process_once(
        lambda rows: [{"bbox": [0, 0, 10, 10], "frame_id": r.value["frame_id"]} for r in rows],
        output_topic="frames.out",
    )

    assert consumed == 2
    assert fake_consumer.commits == 1
    assert len(fake_producer.sent) == 2
    assert fake_producer.sent[0]["topic"] == "frames.out"
    assert fake_producer.sent[0]["key"] == "cam-1"
    assert fake_producer.sent[1]["key"] == "cam-2"


def test_kafka_audit_logger_emits_payload() -> None:
    fake_producer = _FakeKafkaProducer()
    producer = KafkaProducerClient(producer=fake_producer)
    logger = KafkaAuditLogger(producer, topic="ai.audit")

    event = logger.log(
        user_id="user-1",
        model="gpt-4o",
        prompt="Summarize report",
        response="Summary complete",
        request_id="req-abc",
        metadata={"region": "jp"},
        wait=False,
    )

    assert event.user_id == "user-1"
    assert event.request_id == "req-abc"
    assert event.metadata == {"region": "jp"}
    assert event.timestamp_utc

    assert len(fake_producer.sent) == 1
    payload = fake_producer.sent[0]["value"]
    assert payload["user_id"] == "user-1"
    assert payload["model"] == "gpt-4o"
    assert payload["prompt"] == "Summarize report"
    assert payload["response"] == "Summary complete"
    assert payload["request_id"] == "req-abc"
    assert fake_producer.sent[0]["topic"] == "ai.audit"
    assert fake_producer.sent[0]["key"] == "user-1"
