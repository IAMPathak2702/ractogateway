# Kafka

The `kafka` module provides typed wrappers for durable event publishing, consuming, stream processing, and immutable audit trails.

## Installation

```bash
pip install ractogateway[kafka]
```

## Publishing Events

```python
from ractogateway.kafka import KafkaProducerClient, KafkaProducerConfig

producer = KafkaProducerClient(
    config=KafkaProducerConfig(bootstrap_servers="localhost:9092")
)

producer.publish("agent.events", {"agent": "research", "status": "ready"})
```

## Consuming Events

```python
from ractogateway.kafka import KafkaConsumerClient, KafkaConsumerConfig

consumer = KafkaConsumerClient(
    config=KafkaConsumerConfig(
        topics=["agent.events"],
        bootstrap_servers="localhost:9092",
        group_id="writer-agent",
    )
)

messages = consumer.poll(timeout_ms=1000)
for msg in messages:
    print(msg.key, msg.value)
```

## Stream Processing

`KafkaStreamProcessor` consumes from one topic, transforms each message, and publishes to another:

```python
from ractogateway.kafka import KafkaStreamProcessor, KafkaStreamConfig

def transform(msg):
    # Call your LLM kit here, return enriched payload
    return {"summary": "...", "original": msg.value}

processor = KafkaStreamProcessor(
    config=KafkaStreamConfig(
        input_topic="raw.documents",
        output_topic="processed.summaries",
        bootstrap_servers="localhost:9092",
        group_id="summary-processor",
    ),
    transform_fn=transform,
)

processor.run(batch_size=10)
```

## Audit Logging

`KafkaAuditLogger` writes immutable prompt/response records to a dedicated audit topic:

```python
from ractogateway.kafka import KafkaAuditLogger

audit = KafkaAuditLogger(producer, topic="ai.audit")

audit.log(
    user_id="u-42",
    model="gpt-4o",
    prompt="Summarize incident report",
    response="Summary …",
)
```

Audit events are stored as `KafkaAuditEvent` Pydantic models with timestamps, making them easy to query and replay.
