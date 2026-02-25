"""RactoTracer — OpenTelemetry integration for RactoGateway.

Pass a ``RactoTracer`` instance as ``tracer=`` to any developer kit to
automatically emit OTEL spans for every LLM call.

Requires: ``pip install ractogateway[telemetry]``

Example::

    from ractogateway import openai_developer_kit as opd
    from ractogateway.telemetry import RactoTracer

    tracer = RactoTracer(
        otlp_endpoint="http://localhost:4317",
        console=True,
    )
    kit = opd.OpenAIDeveloperKit(
        model="gpt-4o",
        default_prompt=my_prompt,
        tracer=tracer,
    )
    response = kit.chat(opd.ChatConfig(user_message="Hello"))
    # A span named "llm.chat" is now in your OTEL backend.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ractogateway.telemetry._models import ModelPricing, SpanRecord
from ractogateway.telemetry._pricing import DEFAULT_COST_TABLE


def _require_otel_sdk() -> Any:
    try:
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "OpenTelemetry SDK is required for RactoTracer. "
            "Install with:  pip install ractogateway[telemetry]"
        ) from exc
    return TracerProvider


def _require_otlp_grpc() -> Any:
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-untyped]
            OTLPSpanExporter,
        )
    except ImportError as exc:
        raise ImportError(
            "OTLP gRPC exporter is required for otlp_endpoint. "
            "Install with:  pip install ractogateway[telemetry]"
        ) from exc
    return OTLPSpanExporter


def _require_otlp_http() -> Any:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-untyped]
            OTLPSpanExporter,
        )
    except ImportError as exc:
        raise ImportError(
            "OTLP HTTP exporter is required for otlp_http_endpoint. "
            "Install with:  pip install ractogateway[telemetry]"
        ) from exc
    return OTLPSpanExporter


class _InMemoryExporter:
    """Pure-Python in-memory span store — no OTEL dependency needed."""

    def __init__(self) -> None:
        self._records: list[SpanRecord] = []
        self._lock: threading.Lock = threading.Lock()

    def add(self, record: SpanRecord) -> None:
        with self._lock:
            self._records.append(record)

    @property
    def records(self) -> list[SpanRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class RactoTracer:
    """OpenTelemetry tracer — pass as ``tracer=`` to any developer kit.

    Records one span per LLM call with attributes for latency, token
    usage, estimated cost, cache-hit type, and tool-call count.

    Supports OTLP gRPC (Jaeger / Grafana Tempo), OTLP HTTP, console
    stdout, in-memory capture (for tests), and any custom
    ``opentelemetry.sdk.trace.export.SpanExporter``.

    Parameters
    ----------
    service_name:
        OTEL ``service.name`` resource attribute.
        Defaults to ``"ractogateway"``.
    otlp_endpoint:
        OTLP **gRPC** endpoint (e.g. ``"http://localhost:4317"``).
        Requires ``pip install ractogateway[telemetry]``.
    otlp_http_endpoint:
        OTLP **HTTP** endpoint (e.g. ``"http://localhost:4318"``).
        Requires ``pip install ractogateway[telemetry]``.
    console:
        Also print spans to stdout — convenient during local development.
    in_memory:
        Capture spans internally in a thread-safe list.
        Access recorded spans via the :attr:`spans` property.
        Useful for unit tests — no external backend required.
    custom_exporter:
        Any ``opentelemetry.sdk.trace.export.SpanExporter`` instance.
    price_table:
        Override or extend the built-in :data:`~ractogateway.telemetry.DEFAULT_COST_TABLE`.
        Keys are model identifiers; values are :class:`ModelPricing` objects.

    Span attributes
    ---------------
    All spans carry the following OTEL attributes:

    * ``llm.provider`` — ``"openai"`` / ``"google"`` / ``"anthropic"``
    * ``llm.model`` — e.g. ``"gpt-4o"``
    * ``llm.operation`` — ``"chat"`` / ``"stream"`` / ``"embed"``
    * ``llm.latency_ms`` — wall-clock time in milliseconds
    * ``llm.input_tokens`` — prompt tokens consumed
    * ``llm.output_tokens`` — completion tokens produced
    * ``llm.cost_usd`` — estimated USD cost (8 decimal places)
    * ``llm.cache_hit`` — ``"exact"`` / ``"semantic"`` / ``"miss"``
    * ``llm.tool_calls`` — number of tool calls in the response
    * ``llm.error_type`` — exception class name on error (omitted on success)
    """

    def __init__(
        self,
        *,
        service_name: str = "ractogateway",
        otlp_endpoint: str | None = None,
        otlp_http_endpoint: str | None = None,
        console: bool = False,
        in_memory: bool = False,
        custom_exporter: Any | None = None,
        price_table: dict[str, ModelPricing] | None = None,
    ) -> None:
        self._service_name = service_name
        self._otlp_endpoint = otlp_endpoint
        self._otlp_http_endpoint = otlp_http_endpoint
        self._console = console
        self._in_memory = in_memory
        self._custom_exporter = custom_exporter
        self._price_table: dict[str, ModelPricing] = {
            **DEFAULT_COST_TABLE,
            **(price_table or {}),
        }
        self._in_memory_exporter: _InMemoryExporter | None = (
            _InMemoryExporter() if in_memory else None
        )
        # Build OTEL TracerProvider only when at least one exporter is requested.
        self._otel_tracer: Any | None = None
        if otlp_endpoint or otlp_http_endpoint or console or custom_exporter:
            self._otel_tracer = self._build_otel_tracer()

    # ------------------------------------------------------------------
    # Private — OTEL setup
    # ------------------------------------------------------------------

    def _build_otel_tracer(self) -> Any:
        """Construct and configure an OpenTelemetry ``TracerProvider``."""
        from opentelemetry.sdk.resources import (  # type: ignore[import-untyped]
            SERVICE_NAME,
            Resource,
        )
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-untyped]
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-untyped]
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({SERVICE_NAME: self._service_name})
        provider = TracerProvider(resource=resource)

        if self._otlp_endpoint:
            OTLPGrpc = _require_otlp_grpc()
            provider.add_span_processor(
                BatchSpanProcessor(OTLPGrpc(endpoint=self._otlp_endpoint))
            )

        if self._otlp_http_endpoint:
            OTLPHttp = _require_otlp_http()
            provider.add_span_processor(
                BatchSpanProcessor(OTLPHttp(endpoint=self._otlp_http_endpoint))
            )

        if self._console:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        if self._custom_exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(self._custom_exporter))

        return provider.get_tracer(self._service_name)

    # ------------------------------------------------------------------
    # Private — cost + OTEL span emission
    # ------------------------------------------------------------------

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self._price_table.get(model)
        if pricing is None:
            return 0.0
        return (
            input_tokens * pricing.input_per_million / 1_000_000
            + output_tokens * pricing.output_per_million / 1_000_000
        )

    def _emit_otel_span(
        self,
        name: str,
        provider: str,
        model: str,
        operation: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cache_hit: str,
        tool_calls: int,
        status: str,
        error_type: str | None,
    ) -> None:
        if self._otel_tracer is None:
            return
        from opentelemetry.trace import StatusCode  # type: ignore[import-untyped]

        with self._otel_tracer.start_as_current_span(name) as span:
            span.set_attribute("llm.provider", provider)
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.operation", operation)
            span.set_attribute("llm.latency_ms", round(latency_ms, 2))
            span.set_attribute("llm.input_tokens", input_tokens)
            span.set_attribute("llm.output_tokens", output_tokens)
            span.set_attribute("llm.cost_usd", round(cost_usd, 8))
            span.set_attribute("llm.cache_hit", cache_hit)
            span.set_attribute("llm.tool_calls", tool_calls)
            if status == "error":
                span.set_status(StatusCode.ERROR, error_type or "unknown")
            if error_type:
                span.set_attribute("llm.error_type", error_type)

    # ------------------------------------------------------------------
    # Public recording API (called by developer kits)
    # ------------------------------------------------------------------

    def record_chat_span(
        self,
        *,
        provider: str,
        model: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: str = "miss",
        tool_calls: int = 0,
        status: str = "ok",
        error_type: str | None = None,
    ) -> None:
        """Record a completed chat or stream span.

        Parameters
        ----------
        provider:
            Provider string (``"openai"``, ``"google"``, ``"anthropic"``).
        model:
            Model identifier (e.g. ``"gpt-4o"``).
        latency_ms:
            Total wall-clock latency of the LLM call in milliseconds.
        input_tokens:
            Number of prompt tokens consumed (``0`` for cache hits).
        output_tokens:
            Number of completion tokens produced (``0`` for cache hits).
        cache_hit:
            ``"exact"``, ``"semantic"``, or ``"miss"``.
        tool_calls:
            Number of tool calls in the response.
        status:
            ``"ok"`` or ``"error"``.
        error_type:
            Exception class name when ``status == "error"``, else ``None``.
        """
        cost_usd = self._compute_cost(model, input_tokens, output_tokens)
        self._emit_otel_span(
            name="llm.chat",
            provider=provider,
            model=model,
            operation="chat",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cache_hit=cache_hit,
            tool_calls=tool_calls,
            status=status,
            error_type=error_type,
        )
        if self._in_memory_exporter is not None:
            self._in_memory_exporter.add(
                SpanRecord(
                    name="llm.chat",
                    provider=provider,
                    model=model,
                    operation="chat",
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    cache_hit=cache_hit,
                    tool_calls=tool_calls,
                    status=status,
                    error_type=error_type,
                    timestamp=time.time(),
                )
            )

    def record_embed_span(
        self,
        *,
        provider: str,
        model: str,
        latency_ms: float,
        input_tokens: int = 0,
        status: str = "ok",
        error_type: str | None = None,
    ) -> None:
        """Record a completed embedding span.

        Parameters
        ----------
        provider:
            Provider string (``"openai"`` or ``"google"``).
        model:
            Embedding model identifier.
        latency_ms:
            Total wall-clock latency in milliseconds.
        input_tokens:
            Number of tokens embedded.
        status:
            ``"ok"`` or ``"error"``.
        error_type:
            Exception class name when ``status == "error"``, else ``None``.
        """
        cost_usd = self._compute_cost(model, input_tokens, 0)
        self._emit_otel_span(
            name="llm.embed",
            provider=provider,
            model=model,
            operation="embed",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=0,
            cost_usd=cost_usd,
            cache_hit="miss",
            tool_calls=0,
            status=status,
            error_type=error_type,
        )
        if self._in_memory_exporter is not None:
            self._in_memory_exporter.add(
                SpanRecord(
                    name="llm.embed",
                    provider=provider,
                    model=model,
                    operation="embed",
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=0,
                    cost_usd=cost_usd,
                    cache_hit="miss",
                    tool_calls=0,
                    status=status,
                    error_type=error_type,
                    timestamp=time.time(),
                )
            )

    # ------------------------------------------------------------------
    # In-memory span access (for tests)
    # ------------------------------------------------------------------

    @property
    def spans(self) -> list[SpanRecord]:
        """Return all captured in-memory spans.

        Only populated when ``in_memory=True``.  Thread-safe.

        Returns
        -------
        list[SpanRecord]
            Snapshot of all recorded spans (newest last).
        """
        if self._in_memory_exporter is None:
            return []
        return self._in_memory_exporter.records

    def clear_spans(self) -> None:
        """Clear all in-memory spans.

        Only has effect when ``in_memory=True``.
        """
        if self._in_memory_exporter is not None:
            self._in_memory_exporter.clear()
