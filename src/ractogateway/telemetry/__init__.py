"""RactoGateway Telemetry — OpenTelemetry tracing + Prometheus metrics.

Provides production-grade observability for every LLM call made through
any RactoGateway developer kit.

Quick start::

    from ractogateway import openai_developer_kit as opd
    from ractogateway.telemetry import RactoTracer, GatewayMetricsMiddleware, PrometheusExporter

    # --- OTEL tracing ---
    tracer = RactoTracer(otlp_endpoint="http://localhost:4317")

    # --- Prometheus metrics ---
    metrics = GatewayMetricsMiddleware()
    PrometheusExporter(port=8000).start()

    kit = opd.OpenAIDeveloperKit(
        model="gpt-4o",
        default_prompt=my_prompt,
        tracer=tracer,
        metrics=metrics,
    )
    response = kit.chat(opd.ChatConfig(user_message="Hello"))

Install extras::

    pip install ractogateway[telemetry]    # OpenTelemetry tracing
    pip install ractogateway[prometheus]   # Prometheus metrics
    pip install ractogateway[observability]  # both

Public API
----------
:class:`RactoTracer`
    OpenTelemetry tracer.  Pass as ``tracer=`` to any kit.
:class:`GatewayMetricsMiddleware`
    Prometheus metrics collector.  Pass as ``metrics=`` to any kit.
:class:`PrometheusExporter`
    HTTP server for Prometheus scraping.
:class:`ModelPricing`
    Per-model input/output pricing (USD per 1M tokens).
:class:`SpanRecord`
    In-memory span record for test assertions.
:data:`DEFAULT_COST_TABLE`
    Built-in pricing table.
:func:`compute_cost`
    Compute estimated USD cost from token counts.
"""

from ractogateway.telemetry._models import ModelPricing, SpanRecord
from ractogateway.telemetry._pricing import DEFAULT_COST_TABLE, compute_cost
from ractogateway.telemetry.metrics import GatewayMetricsMiddleware
from ractogateway.telemetry.prometheus_exporter import PrometheusExporter
from ractogateway.telemetry.tracer import RactoTracer

__all__ = [
    "DEFAULT_COST_TABLE",
    "GatewayMetricsMiddleware",
    "ModelPricing",
    "PrometheusExporter",
    "RactoTracer",
    "SpanRecord",
    "compute_cost",
]
