"""Kafka streaming integration for RactoGateway.

Four production patterns exposed through typed wrappers:

* :class:`KafkaProducerClient` for durable event publishing.
* :class:`KafkaConsumerClient` for typed polling.
* :class:`KafkaStreamProcessor` for batched consume/transform/publish loops.
* :class:`KafkaAuditLogger` for immutable prompt/response audit trails.

Quick start::

    pip install ractogateway[kafka]

    from ractogateway.kafka import (
        KafkaProducerClient,
        KafkaConsumerClient,
        KafkaProducerConfig,
        KafkaConsumerConfig,
        KafkaStreamProcessor,
        KafkaAuditLogger,
    )

    producer = KafkaProducerClient(
        config=KafkaProducerConfig(bootstrap_servers="localhost:9092")
    )
    producer.publish("agent.events", {"agent": "research", "status": "ready"})

    consumer = KafkaConsumerClient(
        config=KafkaConsumerConfig(
            topics=["agent.events"],
            bootstrap_servers="localhost:9092",
            group_id="writer-agent",
        )
    )
    messages = consumer.poll(timeout_ms=1000)

    audit = KafkaAuditLogger(producer, topic="ai.audit")
    audit.log(
        user_id="u-42",
        model="gpt-4o",
        prompt="Summarize incident report",
        response="Summary ...",
    )
"""

from ractogateway.kafka._models import (
    KafkaAuditEvent,
    KafkaConsumerConfig,
    KafkaMessage,
    KafkaProducerConfig,
    KafkaProduceResult,
    KafkaPublishRequest,
    KafkaStreamConfig,
)
from ractogateway.kafka.audit import KafkaAuditLogger
from ractogateway.kafka.consumer import KafkaConsumerClient
from ractogateway.kafka.producer import KafkaProducerClient
from ractogateway.kafka.stream import KafkaStreamProcessor

__all__ = [
    "KafkaAuditEvent",
    "KafkaAuditLogger",
    "KafkaConsumerClient",
    "KafkaConsumerConfig",
    "KafkaMessage",
    "KafkaProduceResult",
    "KafkaProducerClient",
    "KafkaProducerConfig",
    "KafkaPublishRequest",
    "KafkaStreamConfig",
    "KafkaStreamProcessor",
]

