"""Immutable prompt/response audit logging over Kafka."""

from __future__ import annotations

from typing import Any

from ractogateway.kafka._models import KafkaAuditEvent
from ractogateway.kafka.producer import KafkaProducerClient


class KafkaAuditLogger:
    """Asynchronous audit logger for regulated AI workloads.

    Writes immutable prompt/response records to a dedicated Kafka topic so
    audit traces survive web-process crashes and can be replayed later.
    """

    def __init__(
        self,
        producer: KafkaProducerClient,
        *,
        topic: str = "ractogateway.audit",
    ) -> None:
        self._producer = producer
        self._topic = topic

    @property
    def topic(self) -> str:
        return self._topic

    def log(
        self,
        *,
        user_id: str,
        model: str,
        prompt: str,
        response: str,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        wait: bool = False,
    ) -> KafkaAuditEvent:
        """Publish one immutable audit record.

        Parameters
        ----------
        wait:
            If ``True``, block until Kafka acknowledges the write. For
            high-throughput request paths, ``False`` is usually preferred.
        """
        event = KafkaAuditEvent(
            user_id=user_id,
            model=model,
            prompt=prompt,
            response=response,
            request_id=request_id,
            metadata=metadata or {},
        )
        self._producer.publish(
            self._topic,
            event.model_dump(),
            key=user_id,
            wait=wait,
        )
        return event

