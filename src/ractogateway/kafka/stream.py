"""Batched consume-process-publish utilities for real-time Kafka streams."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from ractogateway.kafka._models import KafkaMessage, KafkaPublishRequest, KafkaStreamConfig
from ractogateway.kafka.consumer import KafkaConsumerClient
from ractogateway.kafka.producer import KafkaProducerClient

StreamHandler = Callable[[list[KafkaMessage]], Sequence[Any] | None]


class KafkaStreamProcessor:
    """Real-time stream loop helper for high-throughput event processing.

    Parameters
    ----------
    consumer:
        Source consumer used to read input events.
    producer:
        Optional producer used to publish handler outputs.
    config:
        Batch and commit controls. Defaults are applied when omitted.
    """

    def __init__(
        self,
        *,
        consumer: KafkaConsumerClient,
        producer: KafkaProducerClient | None = None,
        config: KafkaStreamConfig | None = None,
    ) -> None:
        self._consumer = consumer
        self._producer = producer
        self._config = config or KafkaStreamConfig()

    def collect_batch(self) -> list[KafkaMessage]:
        """Collect up to ``batch_size`` messages bounded by ``max_wait_ms``."""
        cfg = self._config
        if cfg.max_wait_ms == 0:
            return self._consumer.poll(
                timeout_ms=cfg.poll_timeout_ms,
                max_records=cfg.batch_size,
            )

        deadline = time.monotonic() + (cfg.max_wait_ms / 1000.0)
        out: list[KafkaMessage] = []

        while len(out) < cfg.batch_size:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            timeout = min(cfg.poll_timeout_ms, remaining_ms)
            out.extend(
                self._consumer.poll(
                    timeout_ms=timeout,
                    max_records=cfg.batch_size - len(out),
                )
            )
            if len(out) >= cfg.batch_size or time.monotonic() >= deadline:
                break
        return out

    def process_once(
        self,
        handler: StreamHandler,
        *,
        output_topic: str | None = None,
    ) -> int:
        """Run one batch cycle and return the number of consumed messages.

        Handler return options
        ----------------------
        * ``None``: consume-only, no output publish.
        * ``Sequence[KafkaPublishRequest]``: explicit per-message publish plans.
        * ``Sequence[Any]``: publish raw values to ``output_topic``.
        """
        batch = self.collect_batch()
        if not batch:
            return 0

        outputs = handler(batch)
        if outputs is not None:
            if self._producer is None:
                raise RuntimeError(
                    "A producer is required when handler returns output events."
                )
            self._publish_outputs(batch, outputs, output_topic=output_topic)

        if self._config.commit_on_success:
            self._consumer.commit()
        return len(batch)

    def run(
        self,
        handler: StreamHandler,
        *,
        output_topic: str | None = None,
        iterations: int | None = None,
    ) -> int:
        """Run repeated batch cycles. Returns total consumed message count."""
        total = 0
        i = 0
        while iterations is None or i < iterations:
            total += self.process_once(handler, output_topic=output_topic)
            i += 1
        return total

    def _publish_outputs(
        self,
        batch: list[KafkaMessage],
        outputs: Sequence[Any],
        *,
        output_topic: str | None,
    ) -> None:
        producer = self._producer
        if producer is None:  # pragma: no cover
            raise RuntimeError("Internal error: producer unexpectedly missing.")

        for idx, payload in enumerate(outputs):
            if isinstance(payload, KafkaPublishRequest):
                producer.publish(
                    payload.topic,
                    payload.value,
                    key=payload.key,
                    headers=payload.headers,
                    partition=payload.partition,
                    timestamp_ms=payload.timestamp_ms,
                    wait=self._config.wait_for_publish,
                )
                continue

            if output_topic is None:
                raise ValueError(
                    "output_topic is required when handler returns raw values."
                )
            key = batch[idx].key if idx < len(batch) else None
            producer.publish(
                output_topic,
                payload,
                key=key,
                wait=self._config.wait_for_publish,
            )

