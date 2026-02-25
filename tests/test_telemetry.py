"""Unit tests for ractogateway.telemetry.

All tests run without real API keys and without an external OTEL/Prometheus backend.
RactoTracer(in_memory=True) captures spans locally.
GatewayMetricsMiddleware uses an isolated CollectorRegistry so tests are independent.

Run with:
    pip install "ractogateway[observability]"
    pytest tests/test_telemetry.py -v
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Optional-dependency detection
# ---------------------------------------------------------------------------
try:
    import prometheus_client  # noqa: F401
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

needs_prometheus = pytest.mark.skipif(
    not HAS_PROMETHEUS,
    reason="pip install ractogateway[prometheus]",
)


# ===========================================================================
# RactoTracer — in-memory mode (no OTEL SDK required)
# ===========================================================================

class TestRactoTracerInMemory:
    """Tests for RactoTracer using in_memory=True (no external backend)."""

    def setup_method(self) -> None:
        from ractogateway.telemetry import RactoTracer
        self.tracer = RactoTracer(in_memory=True)

    def test_initial_state_empty(self) -> None:
        assert self.tracer.spans == []

    def test_record_chat_span_fields(self) -> None:
        self.tracer.record_chat_span(
            provider="openai",
            model="gpt-4o",
            latency_ms=123.4,
            input_tokens=100,
            output_tokens=50,
            tool_calls=2,
        )
        spans = self.tracer.spans
        assert len(spans) == 1
        s = spans[0]
        assert s.provider == "openai"
        assert s.model == "gpt-4o"
        assert s.name == "llm.chat"
        assert s.operation == "chat"
        assert s.latency_ms == pytest.approx(123.4)
        assert s.input_tokens == 100
        assert s.output_tokens == 50
        assert s.tool_calls == 2
        assert s.status == "ok"
        assert s.error_type is None

    def test_record_chat_span_cache_hit(self) -> None:
        self.tracer.record_chat_span(
            provider="google",
            model="gemini-2.0-flash",
            latency_ms=1.0,
            cache_hit="exact",
        )
        s = self.tracer.spans[0]
        assert s.cache_hit == "exact"
        assert s.input_tokens == 0
        assert s.output_tokens == 0

    def test_record_chat_span_semantic_hit(self) -> None:
        self.tracer.record_chat_span(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            latency_ms=2.0,
            cache_hit="semantic",
        )
        assert self.tracer.spans[0].cache_hit == "semantic"

    def test_record_chat_span_error(self) -> None:
        self.tracer.record_chat_span(
            provider="openai",
            model="gpt-4o",
            latency_ms=50.0,
            status="error",
            error_type="APIConnectionError",
        )
        s = self.tracer.spans[0]
        assert s.status == "error"
        assert s.error_type == "APIConnectionError"

    def test_record_embed_span_fields(self) -> None:
        self.tracer.record_embed_span(
            provider="openai",
            model="text-embedding-3-small",
            latency_ms=45.0,
            input_tokens=300,
        )
        spans = self.tracer.spans
        assert len(spans) == 1
        s = spans[0]
        assert s.name == "llm.embed"
        assert s.operation == "embed"
        assert s.input_tokens == 300
        assert s.output_tokens == 0
        assert s.cache_hit == "miss"

    def test_multiple_spans_accumulate(self) -> None:
        self.tracer.record_chat_span(provider="openai", model="gpt-4o", latency_ms=10.0)
        self.tracer.record_chat_span(provider="anthropic", model="claude-opus-4-6", latency_ms=20.0)
        self.tracer.record_embed_span(provider="openai", model="text-embedding-3-small",
                                      latency_ms=5.0)
        assert len(self.tracer.spans) == 3

    def test_clear_spans(self) -> None:
        self.tracer.record_chat_span(provider="openai", model="gpt-4o", latency_ms=10.0)
        assert len(self.tracer.spans) == 1
        self.tracer.clear_spans()
        assert self.tracer.spans == []

    def test_spans_returns_snapshot(self) -> None:
        """Mutating the returned list must not affect the internal store."""
        self.tracer.record_chat_span(provider="openai", model="gpt-4o", latency_ms=10.0)
        snapshot = self.tracer.spans
        snapshot.clear()
        assert len(self.tracer.spans) == 1


# ===========================================================================
# Cost computation
# ===========================================================================

class TestCostComputation:
    def test_known_model_cost(self) -> None:
        from ractogateway.telemetry import compute_cost
        # gpt-4o: $2.50/1M input, $10.00/1M output
        cost = compute_cost("gpt-4o", 1_000_000, 0)
        assert cost == pytest.approx(2.50, rel=1e-3)

    def test_unknown_model_returns_zero(self) -> None:
        from ractogateway.telemetry import compute_cost
        assert compute_cost("unknown-model-xyz", 100, 100) == 0.0

    def test_combined_input_output_cost(self) -> None:
        from ractogateway.telemetry import compute_cost
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = compute_cost("gpt-4o-mini", 500_000, 500_000)
        expected = 0.15 * 0.5 + 0.60 * 0.5
        assert cost == pytest.approx(expected, rel=1e-3)

    def test_cost_stored_on_span(self) -> None:
        from ractogateway.telemetry import RactoTracer
        tracer = RactoTracer(in_memory=True)
        tracer.record_chat_span(
            provider="openai",
            model="gpt-4o",
            latency_ms=10.0,
            input_tokens=1_000_000,
            output_tokens=0,
        )
        assert tracer.spans[0].cost_usd == pytest.approx(2.50, rel=1e-3)

    def test_custom_price_table_overrides_default(self) -> None:
        from ractogateway.telemetry import ModelPricing, RactoTracer
        custom = {"my-custom-model": ModelPricing(input_per_million=1.0, output_per_million=2.0)}
        tracer = RactoTracer(in_memory=True, price_table=custom)
        tracer.record_chat_span(
            provider="openai",
            model="my-custom-model",
            latency_ms=5.0,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert tracer.spans[0].cost_usd == pytest.approx(3.0, rel=1e-3)

    def test_default_cost_table_contains_all_providers(self) -> None:
        from ractogateway.telemetry import DEFAULT_COST_TABLE
        models = set(DEFAULT_COST_TABLE.keys())
        assert any("gpt" in m for m in models), "OpenAI models missing"
        assert any("gemini" in m for m in models), "Google models missing"
        assert any("claude" in m for m in models), "Anthropic models missing"


# ===========================================================================
# SpanRecord model
# ===========================================================================

class TestSpanRecord:
    def test_defaults(self) -> None:
        from ractogateway.telemetry import SpanRecord
        s = SpanRecord(
            name="llm.chat",
            provider="openai",
            model="gpt-4o",
            operation="chat",
            latency_ms=100.0,
        )
        assert s.input_tokens == 0
        assert s.output_tokens == 0
        assert s.cost_usd == 0.0
        assert s.cache_hit == "miss"
        assert s.tool_calls == 0
        assert s.status == "ok"
        assert s.error_type is None
        assert s.timestamp > 0

    def test_pydantic_validation_rejects_negative_latency(self) -> None:
        from pydantic import ValidationError
        from ractogateway.telemetry import SpanRecord
        with pytest.raises(ValidationError):
            SpanRecord(
                name="llm.chat",
                provider="openai",
                model="gpt-4o",
                operation="chat",
                latency_ms=-1.0,
            )


# ===========================================================================
# GatewayMetricsMiddleware (requires prometheus_client)
# ===========================================================================

@needs_prometheus
class TestGatewayMetricsMiddleware:
    """Prometheus metrics tests — isolated CollectorRegistry per instance."""

    def _make_middleware(self) -> "GatewayMetricsMiddleware":  # type: ignore[name-defined]
        import prometheus_client
        from ractogateway.telemetry import GatewayMetricsMiddleware
        registry = prometheus_client.CollectorRegistry()
        return GatewayMetricsMiddleware(registry=registry)

    def test_requests_total_incremented(self) -> None:
        m = self._make_middleware()
        m.record_request(
            provider="openai", model="gpt-4o", operation="chat",
            status="ok", latency_s=0.5, input_tokens=100, output_tokens=50,
        )
        text = m.generate_latest()
        assert "ractogateway_requests_total" in text

    def test_duration_histogram_observed(self) -> None:
        m = self._make_middleware()
        m.record_request(
            provider="openai", model="gpt-4o", operation="chat",
            status="ok", latency_s=1.23,
        )
        text = m.generate_latest()
        assert "ractogateway_request_duration_seconds" in text

    def test_tokens_total_incremented(self) -> None:
        m = self._make_middleware()
        m.record_request(
            provider="anthropic", model="claude-haiku-4-5-20251001",
            operation="chat", status="ok", latency_s=0.2,
            input_tokens=200, output_tokens=100,
        )
        text = m.generate_latest()
        assert "ractogateway_tokens_total" in text

    def test_cost_counter_updated(self) -> None:
        m = self._make_middleware()
        m.record_request(
            provider="openai", model="gpt-4o", operation="chat",
            status="ok", latency_s=0.5, input_tokens=1_000_000, output_tokens=0,
        )
        text = m.generate_latest()
        assert "ractogateway_cost_usd_total" in text

    def test_cache_hit_counter(self) -> None:
        m = self._make_middleware()
        m.record_cache_hit("exact")
        m.record_cache_hit("exact")
        m.record_cache_hit("semantic")
        text = m.generate_latest()
        assert "ractogateway_cache_hits_total" in text

    def test_cache_miss_counter(self) -> None:
        m = self._make_middleware()
        m.record_cache_miss("exact")
        text = m.generate_latest()
        assert "ractogateway_cache_misses_total" in text

    def test_tool_calls_counter(self) -> None:
        from ractogateway.adapters.base import ToolCallResult
        m = self._make_middleware()
        tc = ToolCallResult(id="call_1", name="search", arguments={})
        m.record_request(
            provider="openai", model="gpt-4o", operation="chat",
            status="ok", latency_s=0.5, tool_calls=[tc],
        )
        text = m.generate_latest()
        assert "ractogateway_tool_calls_total" in text

    def test_error_status_recorded(self) -> None:
        m = self._make_middleware()
        m.record_request(
            provider="openai", model="gpt-4o", operation="chat",
            status="error", latency_s=0.1,
        )
        text = m.generate_latest()
        assert 'status="error"' in text

    def test_generate_latest_returns_string(self) -> None:
        m = self._make_middleware()
        result = m.generate_latest()
        assert isinstance(result, str)

    def test_isolated_registries_do_not_interfere(self) -> None:
        m1 = self._make_middleware()
        m2 = self._make_middleware()
        m1.record_request(
            provider="openai", model="gpt-4o", operation="chat",
            status="ok", latency_s=0.5,
        )
        # m2 should have zero requests
        text2 = m2.generate_latest()
        assert "openai" not in text2 or 'ractogateway_requests_total{' not in text2


# ===========================================================================
# PrometheusExporter lifecycle (requires prometheus_client)
# ===========================================================================

@needs_prometheus
class TestPrometheusExporter:
    def test_start_and_stop(self) -> None:
        from ractogateway.telemetry import PrometheusExporter
        exp = PrometheusExporter(port=19876)
        exp.start()
        assert exp.is_running
        exp.stop()
        assert not exp.is_running

    def test_double_start_is_idempotent(self) -> None:
        from ractogateway.telemetry import PrometheusExporter
        exp = PrometheusExporter(port=19877)
        exp.start()
        exp.start()  # second call is a no-op
        assert exp.is_running
        exp.stop()

    def test_stop_without_start_is_safe(self) -> None:
        from ractogateway.telemetry import PrometheusExporter
        exp = PrometheusExporter(port=19878)
        exp.stop()  # should not raise
        assert not exp.is_running


# ===========================================================================
# Kit integration — OpenAI (mocked adapter)
# ===========================================================================

class TestOpenAIKitTelemetry:
    """Verify that OpenAIDeveloperKit correctly wires tracer + metrics.

    The underlying adapter.run() is mocked so no real API call is made.
    """

    def _make_response(
        self,
        content: str = "Hello!",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
    ) -> "LLMResponse":  # type: ignore[name-defined]
        from ractogateway.adapters.base import FinishReason, LLMResponse
        return LLMResponse(
            content=content,
            finish_reason=FinishReason.STOP,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                   "total_tokens": prompt_tokens + completion_tokens},
        )

    def test_chat_records_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from ractogateway import openai_developer_kit as opd
        from ractogateway.telemetry import RactoTracer
        from ractogateway.prompts.engine import RactoPrompt

        tracer = RactoTracer(in_memory=True)
        prompt = RactoPrompt(
            context="You are helpful.",
            instructions="Answer clearly.",
            output_format="Return a plain text answer.",
        )
        kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt, tracer=tracer)

        mock_resp = self._make_response()
        monkeypatch.setattr(kit._adapter, "run", MagicMock(return_value=mock_resp))

        kit.chat(opd.ChatConfig(user_message="Hello"))

        spans = tracer.spans
        assert len(spans) == 1
        s = spans[0]
        assert s.provider == "openai"
        assert s.model == "gpt-4o"
        assert s.operation == "chat"
        assert s.status == "ok"
        assert s.input_tokens == 10
        assert s.output_tokens == 5
        assert s.latency_ms >= 0

    def test_chat_records_error_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from ractogateway import openai_developer_kit as opd
        from ractogateway.telemetry import RactoTracer
        from ractogateway.prompts.engine import RactoPrompt

        tracer = RactoTracer(in_memory=True)
        prompt = RactoPrompt(
            context="You are helpful.",
            instructions="Answer clearly.",
            output_format="Return a plain text answer.",
        )
        kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt, tracer=tracer)

        monkeypatch.setattr(
            kit._adapter, "run", MagicMock(side_effect=RuntimeError("network error"))
        )

        with pytest.raises(RuntimeError):
            kit.chat(opd.ChatConfig(user_message="Hello"))

        assert len(tracer.spans) == 1
        assert tracer.spans[0].status == "error"
        assert tracer.spans[0].error_type == "RuntimeError"

    @needs_prometheus
    def test_chat_records_prometheus_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import prometheus_client
        from unittest.mock import MagicMock
        from ractogateway import openai_developer_kit as opd
        from ractogateway.telemetry import GatewayMetricsMiddleware
        from ractogateway.prompts.engine import RactoPrompt

        registry = prometheus_client.CollectorRegistry()
        metrics = GatewayMetricsMiddleware(registry=registry)
        prompt = RactoPrompt(
            context="You are helpful.",
            instructions="Answer clearly.",
            output_format="Return a plain text answer.",
        )
        kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt, metrics=metrics)

        mock_resp = self._make_response()
        monkeypatch.setattr(kit._adapter, "run", MagicMock(return_value=mock_resp))
        kit.chat(opd.ChatConfig(user_message="Hello"))

        text = metrics.generate_latest()
        assert "ractogateway_requests_total" in text
        assert "ractogateway_request_duration_seconds" in text


# ===========================================================================
# Kit integration — Google (mocked adapter)
# ===========================================================================

class TestGoogleKitTelemetry:
    def _make_response(self) -> "LLMResponse":  # type: ignore[name-defined]
        from ractogateway.adapters.base import FinishReason, LLMResponse
        return LLMResponse(
            content="Hi there!",
            finish_reason=FinishReason.STOP,
            usage={"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        )

    def test_chat_records_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from ractogateway import google_developer_kit as god
        from ractogateway.telemetry import RactoTracer
        from ractogateway.prompts.engine import RactoPrompt

        tracer = RactoTracer(in_memory=True)
        prompt = RactoPrompt(
            context="You are helpful.",
            instructions="Answer clearly.",
            output_format="Return a plain text answer.",
        )
        kit = god.GoogleDeveloperKit(model="gemini-2.0-flash", default_prompt=prompt,
                                     tracer=tracer)

        mock_resp = self._make_response()
        monkeypatch.setattr(kit._adapter, "run", MagicMock(return_value=mock_resp))

        kit.chat(god.ChatConfig(user_message="Hello"))

        spans = tracer.spans
        assert len(spans) == 1
        s = spans[0]
        assert s.provider == "google"
        assert s.model == "gemini-2.0-flash"
        assert s.status == "ok"
        assert s.input_tokens == 8
        assert s.output_tokens == 4

    def test_chat_records_error_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from ractogateway import google_developer_kit as god
        from ractogateway.telemetry import RactoTracer
        from ractogateway.prompts.engine import RactoPrompt

        tracer = RactoTracer(in_memory=True)
        prompt = RactoPrompt(
            context="You are helpful.",
            instructions="Answer clearly.",
            output_format="Return a plain text answer.",
        )
        kit = god.GoogleDeveloperKit(model="gemini-2.0-flash", default_prompt=prompt,
                                     tracer=tracer)
        monkeypatch.setattr(
            kit._adapter, "run", MagicMock(side_effect=ConnectionError("timeout"))
        )

        with pytest.raises(ConnectionError):
            kit.chat(god.ChatConfig(user_message="Hello"))

        assert tracer.spans[0].status == "error"
        assert tracer.spans[0].error_type == "ConnectionError"


# ===========================================================================
# Kit integration — Anthropic (mocked adapter)
# ===========================================================================

class TestAnthropicKitTelemetry:
    def _make_response(self) -> "LLMResponse":  # type: ignore[name-defined]
        from ractogateway.adapters.base import FinishReason, LLMResponse
        return LLMResponse(
            content="Sure!",
            finish_reason=FinishReason.STOP,
            usage={"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
        )

    def test_chat_records_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from ractogateway import anthropic_developer_kit as anth
        from ractogateway.telemetry import RactoTracer
        from ractogateway.prompts.engine import RactoPrompt

        tracer = RactoTracer(in_memory=True)
        prompt = RactoPrompt(
            context="You are helpful.",
            instructions="Answer clearly.",
            output_format="Return a plain text answer.",
        )
        kit = anth.AnthropicDeveloperKit(
            model="claude-haiku-4-5-20251001",
            default_prompt=prompt,
            tracer=tracer,
        )

        mock_resp = self._make_response()
        monkeypatch.setattr(kit._adapter, "run", MagicMock(return_value=mock_resp))

        kit.chat(anth.ChatConfig(user_message="Hello"))

        spans = tracer.spans
        assert len(spans) == 1
        s = spans[0]
        assert s.provider == "anthropic"
        assert s.model == "claude-haiku-4-5-20251001"
        assert s.status == "ok"
        assert s.input_tokens == 12
        assert s.output_tokens == 6

    def test_chat_error_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from ractogateway import anthropic_developer_kit as anth
        from ractogateway.telemetry import RactoTracer
        from ractogateway.prompts.engine import RactoPrompt

        tracer = RactoTracer(in_memory=True)
        prompt = RactoPrompt(
            context="You are helpful.",
            instructions="Answer clearly.",
            output_format="Return a plain text answer.",
        )
        kit = anth.AnthropicDeveloperKit(
            model="claude-haiku-4-5-20251001",
            default_prompt=prompt,
            tracer=tracer,
        )
        monkeypatch.setattr(
            kit._adapter, "run", MagicMock(side_effect=ValueError("bad request"))
        )

        with pytest.raises(ValueError):
            kit.chat(anth.ChatConfig(user_message="Hello"))

        assert tracer.spans[0].status == "error"

    @needs_prometheus
    def test_anthropic_prometheus_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import prometheus_client
        from unittest.mock import MagicMock
        from ractogateway import anthropic_developer_kit as anth
        from ractogateway.telemetry import GatewayMetricsMiddleware
        from ractogateway.prompts.engine import RactoPrompt

        registry = prometheus_client.CollectorRegistry()
        metrics = GatewayMetricsMiddleware(registry=registry)
        prompt = RactoPrompt(
            context="You are helpful.",
            instructions="Answer clearly.",
            output_format="Return a plain text answer.",
        )
        kit = anth.AnthropicDeveloperKit(
            model="claude-haiku-4-5-20251001",
            default_prompt=prompt,
            metrics=metrics,
        )
        mock_resp = self._make_response()
        monkeypatch.setattr(kit._adapter, "run", MagicMock(return_value=mock_resp))
        kit.chat(anth.ChatConfig(user_message="Hello"))

        text = metrics.generate_latest()
        assert "ractogateway_requests_total" in text
        assert "anthropic" in text


# ===========================================================================
# Shared tracer across multiple kits
# ===========================================================================

class TestSharedTracer:
    """One tracer instance aggregating spans from multiple providers."""

    def test_multi_provider_spans_accumulate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        from ractogateway import anthropic_developer_kit as anth
        from ractogateway import openai_developer_kit as opd
        from ractogateway.adapters.base import FinishReason, LLMResponse
        from ractogateway.telemetry import RactoTracer
        from ractogateway.prompts.engine import RactoPrompt

        tracer = RactoTracer(in_memory=True)
        prompt = RactoPrompt(
            context="You are helpful.",
            instructions="Answer clearly.",
            output_format="Return a plain text answer.",
        )

        mock_resp = LLMResponse(
            content="Hello!",
            finish_reason=FinishReason.STOP,
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )

        openai_kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt,
                                            tracer=tracer)
        anth_kit = anth.AnthropicDeveloperKit(
            model="claude-haiku-4-5-20251001", default_prompt=prompt, tracer=tracer
        )

        monkeypatch.setattr(openai_kit._adapter, "run", MagicMock(return_value=mock_resp))
        monkeypatch.setattr(anth_kit._adapter, "run", MagicMock(return_value=mock_resp))

        openai_kit.chat(opd.ChatConfig(user_message="Q1"))
        anth_kit.chat(anth.ChatConfig(user_message="Q2"))

        spans = tracer.spans
        assert len(spans) == 2
        providers = {s.provider for s in spans}
        assert providers == {"openai", "anthropic"}
