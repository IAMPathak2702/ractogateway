"""GatewayMetricsMiddleware — Prometheus metrics for RactoGateway.

Pass a ``GatewayMetricsMiddleware`` instance as ``metrics=`` to any developer
kit to collect per-request Prometheus metrics.

Requires: ``pip install ractogateway[prometheus]``

Metrics exposed
---------------
* ``ractogateway_requests_total{provider,model,operation,status}`` — Counter
* ``ractogateway_request_duration_seconds{provider,model,operation}`` — Histogram
* ``ractogateway_tokens_total{provider,model,token_type}`` — Counter
* ``ractogateway_cost_usd_total{provider,model}`` — Counter
* ``ractogateway_cache_hits_total{cache_type}`` — Counter
* ``ractogateway_cache_misses_total{cache_type}`` — Counter
* ``ractogateway_tool_calls_total{tool_name}`` — Counter

Example::

    from ractogateway import openai_developer_kit as opd
    from ractogateway.telemetry import GatewayMetricsMiddleware, PrometheusExporter

    metrics = GatewayMetricsMiddleware()
    exporter = PrometheusExporter(port=8000)
    exporter.start()

    kit = opd.OpenAIDeveloperKit(
        model="gpt-4o",
        default_prompt=my_prompt,
        metrics=metrics,
    )
    response = kit.chat(opd.ChatConfig(user_message="Hello"))
    # Scrape http://localhost:8000/metrics in Prometheus.
"""

from __future__ import annotations

import threading
from typing import Any

from ractogateway.telemetry._models import ModelPricing
from ractogateway.telemetry._pricing import DEFAULT_COST_TABLE


def _require_prometheus() -> Any:
    try:
        import prometheus_client  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "prometheus_client is required for GatewayMetricsMiddleware. "
            "Install with:  pip install ractogateway[prometheus]"
        ) from exc
    return prometheus_client


class GatewayMetricsMiddleware:
    """Prometheus metrics middleware — pass as ``metrics=`` to any developer kit.

    A single instance can be shared across multiple kits (different
    providers) to aggregate metrics in one registry.

    Parameters
    ----------
    price_table:
        Override or extend the built-in pricing table used for the
        ``ractogateway_cost_usd_total`` counter.
    registry:
        Custom ``prometheus_client.CollectorRegistry``.  Defaults to the
        global ``REGISTRY`` (which also includes default Python metrics).
        Pass ``prometheus_client.CollectorRegistry()`` to get an isolated
        registry — useful in tests.

    Requires: ``pip install ractogateway[prometheus]``
    """

    def __init__(
        self,
        *,
        price_table: dict[str, ModelPricing] | None = None,
        registry: Any | None = None,
    ) -> None:
        self._price_table: dict[str, ModelPricing] = {
            **DEFAULT_COST_TABLE,
            **(price_table or {}),
        }
        self._registry = registry
        self._lock: threading.Lock = threading.Lock()
        self._metrics: dict[str, Any] = self._build_metrics()

    # ------------------------------------------------------------------
    # Private — metric registration
    # ------------------------------------------------------------------

    def _build_metrics(self) -> dict[str, Any]:
        prom = _require_prometheus()
        kw: dict[str, Any] = {}
        if self._registry is not None:
            kw["registry"] = self._registry

        return {
            "requests_total": prom.Counter(
                "ractogateway_requests_total",
                "Total LLM requests by provider / model / operation / status.",
                ["provider", "model", "operation", "status"],
                **kw,
            ),
            "duration_seconds": prom.Histogram(
                "ractogateway_request_duration_seconds",
                "LLM request wall-clock latency in seconds.",
                ["provider", "model", "operation"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
                **kw,
            ),
            "tokens_total": prom.Counter(
                "ractogateway_tokens_total",
                "Total tokens consumed — labelled by token_type (input / output).",
                ["provider", "model", "token_type"],
                **kw,
            ),
            "cost_usd_total": prom.Counter(
                "ractogateway_cost_usd_total",
                "Total estimated cost in USD.",
                ["provider", "model"],
                **kw,
            ),
            "cache_hits_total": prom.Counter(
                "ractogateway_cache_hits_total",
                "Total cache hits by cache type (exact / semantic).",
                ["cache_type"],
                **kw,
            ),
            "cache_misses_total": prom.Counter(
                "ractogateway_cache_misses_total",
                "Total cache misses by cache type (exact / semantic).",
                ["cache_type"],
                **kw,
            ),
            "tool_calls_total": prom.Counter(
                "ractogateway_tool_calls_total",
                "Total tool calls executed, labelled by tool name.",
                ["tool_name"],
                **kw,
            ),
        }

    # ------------------------------------------------------------------
    # Private — cost helper
    # ------------------------------------------------------------------

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self._price_table.get(model)
        if pricing is None:
            return 0.0
        return (
            input_tokens * pricing.input_per_million / 1_000_000
            + output_tokens * pricing.output_per_million / 1_000_000
        )

    # ------------------------------------------------------------------
    # Public recording API (called by developer kits)
    # ------------------------------------------------------------------

    def record_request(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        status: str,
        latency_s: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: list[Any] | None = None,
    ) -> None:
        """Record metrics for a completed LLM request.

        Parameters
        ----------
        provider:
            Provider string (``"openai"``, ``"google"``, ``"anthropic"``).
        model:
            Model identifier (e.g. ``"gpt-4o"``).
        operation:
            ``"chat"``, ``"stream"``, or ``"embed"``.
        status:
            ``"ok"`` or ``"error"``.
        latency_s:
            Request wall-clock latency **in seconds**.
        input_tokens:
            Prompt tokens consumed (``0`` for cache hits or errors).
        output_tokens:
            Completion tokens produced (``0`` for cache hits or errors).
        tool_calls:
            List of ``ToolCallResult`` objects from the response.
            Used to update :metric:`ractogateway_tool_calls_total`.
        """
        m = self._metrics
        m["requests_total"].labels(
            provider=provider, model=model, operation=operation, status=status
        ).inc()
        m["duration_seconds"].labels(
            provider=provider, model=model, operation=operation
        ).observe(latency_s)

        if input_tokens > 0:
            m["tokens_total"].labels(
                provider=provider, model=model, token_type="input"
            ).inc(input_tokens)
        if output_tokens > 0:
            m["tokens_total"].labels(
                provider=provider, model=model, token_type="output"
            ).inc(output_tokens)

        cost = self._compute_cost(model, input_tokens, output_tokens)
        if cost > 0:
            m["cost_usd_total"].labels(provider=provider, model=model).inc(cost)

        for tc in tool_calls or []:
            name = getattr(tc, "name", None) or "unknown"
            m["tool_calls_total"].labels(tool_name=name).inc()

    def record_cache_hit(self, cache_type: str) -> None:
        """Increment the cache-hits counter.

        Parameters
        ----------
        cache_type:
            ``"exact"`` or ``"semantic"``.
        """
        self._metrics["cache_hits_total"].labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str) -> None:
        """Increment the cache-misses counter.

        Parameters
        ----------
        cache_type:
            ``"exact"`` or ``"semantic"``.
        """
        self._metrics["cache_misses_total"].labels(cache_type=cache_type).inc()

    def generate_latest(self) -> str:
        """Return current metrics in Prometheus text exposition format.

        Useful for testing without starting an HTTP server::

            text = middleware.generate_latest()
            assert "ractogateway_requests_total" in text

        Returns
        -------
        str
            UTF-8 decoded Prometheus text format string.
        """
        prom = _require_prometheus()
        if self._registry is not None:
            raw: bytes = prom.generate_latest(self._registry)
        else:
            raw = prom.generate_latest()
        return raw.decode()
