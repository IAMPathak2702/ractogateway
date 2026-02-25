# Telemetry & Observability

RactoGateway ships a production-grade observability layer with **zero code changes** to your
existing kit calls.  Attach a `RactoTracer` and/or `GatewayMetricsMiddleware` to any developer
kit and every LLM call is automatically instrumented.

## Installation

```bash
# OpenTelemetry tracing only
pip install "ractogateway[telemetry]"

# Prometheus metrics only
pip install "ractogateway[prometheus]"

# Both (recommended for production)
pip install "ractogateway[observability]"
```

## Quick start

```python
from ractogateway import openai_developer_kit as opd
from ractogateway.telemetry import RactoTracer, GatewayMetricsMiddleware, PrometheusExporter
from ractogateway.prompts.engine import RactoPrompt

# --- Tracing ---
tracer = RactoTracer(
    otlp_endpoint="http://localhost:4317",   # Jaeger / Grafana Tempo gRPC
    console=True,                            # also print to stdout (dev)
)

# --- Prometheus metrics ---
metrics = GatewayMetricsMiddleware()
PrometheusExporter(port=8000).start()        # scrape http://localhost:8000/metrics

prompt = RactoPrompt(
    context="You are a helpful assistant.",
    instructions="Answer the user's question.",
    output_format="Return a concise plain-text answer.",
)

kit = opd.OpenAIDeveloperKit(
    model="gpt-4o",
    default_prompt=prompt,
    tracer=tracer,     # attach tracer
    metrics=metrics,   # attach metrics
)

response = kit.chat(opd.ChatConfig(user_message="What is 2 + 2?"))
# One OTEL span is now in your backend, one Prometheus data-point is recorded.
```

The same `tracer=` / `metrics=` parameters work on **all three provider kits**.

---

## RactoTracer — OpenTelemetry spans

### Constructor options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `service_name` | `str` | `"ractogateway"` | OTEL `service.name` resource attribute |
| `otlp_endpoint` | `str \| None` | `None` | OTLP **gRPC** endpoint (e.g. Jaeger, Tempo) |
| `otlp_http_endpoint` | `str \| None` | `None` | OTLP **HTTP** endpoint (e.g. Zipkin) |
| `console` | `bool` | `False` | Print spans to stdout |
| `in_memory` | `bool` | `False` | Capture spans in memory (for tests) |
| `custom_exporter` | `SpanExporter \| None` | `None` | Any OTEL `SpanExporter` |
| `price_table` | `dict[str, ModelPricing] \| None` | `None` | Override / extend built-in pricing |

### Span attributes

Every span carries these OTEL attributes:

| Attribute | Type | Description |
|---|---|---|
| `llm.provider` | `string` | `"openai"` / `"google"` / `"anthropic"` |
| `llm.model` | `string` | Model identifier (e.g. `"gpt-4o"`) |
| `llm.operation` | `string` | `"chat"` / `"stream"` / `"embed"` |
| `llm.latency_ms` | `float` | Wall-clock time in milliseconds |
| `llm.input_tokens` | `int` | Prompt tokens consumed |
| `llm.output_tokens` | `int` | Completion tokens produced |
| `llm.cost_usd` | `float` | Estimated USD cost (8 decimal places) |
| `llm.cache_hit` | `string` | `"exact"` / `"semantic"` / `"miss"` |
| `llm.tool_calls` | `int` | Number of tool calls in the response |
| `llm.error_type` | `string` | Exception class name on error (omitted on success) |

### Exporting to Jaeger / Grafana Tempo

```python
# gRPC (default OTLP port 4317)
tracer = RactoTracer(otlp_endpoint="http://jaeger:4317")

# HTTP (default OTLP port 4318)
tracer = RactoTracer(otlp_http_endpoint="http://tempo:4318")
```

### Using in unit tests

Set `in_memory=True` and inspect `.spans` after each call — no external backend needed.

```python
from ractogateway.telemetry import RactoTracer

tracer = RactoTracer(in_memory=True)
kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt, tracer=tracer)

# ... make a (mocked) call ...

assert len(tracer.spans) == 1
span = tracer.spans[0]
assert span.provider == "openai"
assert span.input_tokens > 0
assert span.cost_usd > 0
tracer.clear_spans()   # reset between test cases
```

---

## GatewayMetricsMiddleware — Prometheus metrics

### Metrics exposed

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ractogateway_requests_total` | Counter | `provider`, `model`, `operation`, `status` | Total LLM requests |
| `ractogateway_request_duration_seconds` | Histogram | `provider`, `model`, `operation` | Wall-clock latency |
| `ractogateway_tokens_total` | Counter | `provider`, `model`, `token_type` | Token consumption |
| `ractogateway_cost_usd_total` | Counter | `provider`, `model` | Estimated USD cost |
| `ractogateway_cache_hits_total` | Counter | `cache_type` | Cache hits by type |
| `ractogateway_cache_misses_total` | Counter | `cache_type` | Cache misses by type |
| `ractogateway_tool_calls_total` | Counter | `tool_name` | Tool calls per function |

### Sharing one instance across multiple kits

```python
metrics = GatewayMetricsMiddleware()
PrometheusExporter(port=8000).start()

openai_kit = opd.OpenAIDeveloperKit(model="gpt-4o",         ..., metrics=metrics)
google_kit = god.GoogleDeveloperKit(model="gemini-2.0-flash", ..., metrics=metrics)
anth_kit   = anth.AnthropicDeveloperKit(model="claude-haiku-4-5-20251001", ..., metrics=metrics)
# All three kits write to the same registry → aggregate dashboards out of the box.
```

### Custom Prometheus registry (for tests)

```python
import prometheus_client
from ractogateway.telemetry import GatewayMetricsMiddleware

registry = prometheus_client.CollectorRegistry()  # isolated
metrics = GatewayMetricsMiddleware(registry=registry)
```

---

## Cost estimation

The built-in pricing table covers 40+ models across all three providers.  You can override or
extend it on either `RactoTracer` or `GatewayMetricsMiddleware`:

```python
from ractogateway.telemetry import ModelPricing, RactoTracer

custom_prices = {
    "my-fine-tuned-gpt4": ModelPricing(input_per_million=5.00, output_per_million=15.00),
}

tracer = RactoTracer(in_memory=True, price_table=custom_prices)
```

The default table is available as `ractogateway.telemetry.DEFAULT_COST_TABLE` and you can
compute one-off costs with `compute_cost(model, input_tokens, output_tokens)`.

---

## Google and Anthropic kits

Both kits accept identical `tracer=` / `metrics=` parameters:

```python
from ractogateway import google_developer_kit as god
from ractogateway import anthropic_developer_kit as anth

google_kit = god.GoogleDeveloperKit(
    model="gemini-2.0-flash",
    default_prompt=prompt,
    tracer=tracer,
    metrics=metrics,
)

anth_kit = anth.AnthropicDeveloperKit(
    model="claude-opus-4-6",
    default_prompt=prompt,
    tracer=tracer,
    metrics=metrics,
)
```

> **Note:** Anthropic does not have a native embedding API, so `record_embed_span` is never
> called by `AnthropicDeveloperKit`.

---

## Combining with caching and routing

Telemetry is fully compatible with all other middleware.  Cache hits are recorded as
`cache_hit="exact"` or `cache_hit="semantic"` — the LLM API is not called and no token costs
are incurred.

```python
from ractogateway.cache import ExactMatchCache
from ractogateway.telemetry import RactoTracer, GatewayMetricsMiddleware

tracer  = RactoTracer(in_memory=True)
metrics = GatewayMetricsMiddleware()
cache   = ExactMatchCache()

kit = opd.OpenAIDeveloperKit(
    model="gpt-4o",
    default_prompt=prompt,
    exact_cache=cache,
    tracer=tracer,
    metrics=metrics,
)
```

After an exact cache hit:

- `tracer.spans[-1].cache_hit == "exact"` — zero tokens recorded
- `metrics` counter `ractogateway_cache_hits_total{cache_type="exact"}` is incremented

---

## PrometheusExporter

```python
from ractogateway.telemetry import PrometheusExporter

exp = PrometheusExporter(port=8000)
exp.start()          # starts a background HTTP daemon thread
print(exp.is_running)  # True

# Prometheus scrapes http://host:8000/metrics automatically.

exp.stop()           # clean shutdown
```

The exporter accepts a custom `registry` parameter if you want to serve only specific metrics:

```python
exp = PrometheusExporter(port=8001, registry=my_registry)
```

---

## See also

- [API reference — telemetry](../api/telemetry.md)
- [Grafana dashboard template](../../dashboards/grafana_dashboard.json)
- [Cache guide](cache.md)
- [Routing guide](routing.md)
