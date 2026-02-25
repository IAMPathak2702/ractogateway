# API Reference — Telemetry

Module: `ractogateway.telemetry`

Install extras: `pip install "ractogateway[observability]"`

---

## RactoTracer

```python
class RactoTracer
```

OpenTelemetry tracer.  Pass as `tracer=` to any developer kit.

### Constructor

```python
RactoTracer(
    *,
    service_name: str = "ractogateway",
    otlp_endpoint: str | None = None,
    otlp_http_endpoint: str | None = None,
    console: bool = False,
    in_memory: bool = False,
    custom_exporter: SpanExporter | None = None,
    price_table: dict[str, ModelPricing] | None = None,
)
```

**Parameters**

- **service_name** (`str`) — OTEL `service.name` resource attribute. Defaults to `"ractogateway"`.
- **otlp_endpoint** (`str | None`) — OTLP gRPC endpoint (e.g. `"http://localhost:4317"`).
  Requires `pip install ractogateway[telemetry]`.
- **otlp_http_endpoint** (`str | None`) — OTLP HTTP endpoint (e.g. `"http://localhost:4318"`).
  Requires `pip install ractogateway[telemetry]`.
- **console** (`bool`) — Also print spans to stdout. Defaults to `False`.
- **in_memory** (`bool`) — Capture spans in a thread-safe list. Access via `.spans`.
  Useful for unit tests. Defaults to `False`.
- **custom_exporter** — Any `opentelemetry.sdk.trace.export.SpanExporter`.
- **price_table** (`dict[str, ModelPricing] | None`) — Override or extend the built-in
  pricing table. Keys are model identifiers; values are `ModelPricing` objects.

### Methods

#### record_chat_span

```python
def record_chat_span(
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
) -> None
```

Record a completed chat or stream span.

**Parameters**

- **provider** — `"openai"`, `"google"`, or `"anthropic"`.
- **model** — Model identifier (e.g. `"gpt-4o"`).
- **latency_ms** — Total wall-clock latency of the LLM call in milliseconds.
- **input_tokens** — Prompt tokens consumed. `0` for cache hits.
- **output_tokens** — Completion tokens produced. `0` for cache hits.
- **cache_hit** — `"exact"`, `"semantic"`, or `"miss"`.
- **tool_calls** — Number of tool calls in the response.
- **status** — `"ok"` or `"error"`.
- **error_type** — Exception class name when `status == "error"`, else `None`.

#### record_embed_span

```python
def record_embed_span(
    *,
    provider: str,
    model: str,
    latency_ms: float,
    input_tokens: int = 0,
    status: str = "ok",
    error_type: str | None = None,
) -> None
```

Record a completed embedding span.

#### spans (property)

```python
@property
def spans(self) -> list[SpanRecord]
```

Return all captured in-memory spans. Only populated when `in_memory=True`. Thread-safe.

#### clear_spans

```python
def clear_spans(self) -> None
```

Clear all in-memory spans. Only has effect when `in_memory=True`.

---

## GatewayMetricsMiddleware

```python
class GatewayMetricsMiddleware
```

Prometheus metrics middleware.  Pass as `metrics=` to any developer kit.

### Constructor

```python
GatewayMetricsMiddleware(
    *,
    price_table: dict[str, ModelPricing] | None = None,
    registry: CollectorRegistry | None = None,
)
```

**Parameters**

- **price_table** — Override or extend the built-in pricing table.
- **registry** — Custom `prometheus_client.CollectorRegistry`. Pass an isolated registry in
  tests to prevent metric name collisions.

### Metrics

| Metric name | Type | Labels |
|---|---|---|
| `ractogateway_requests_total` | Counter | `provider`, `model`, `operation`, `status` |
| `ractogateway_request_duration_seconds` | Histogram | `provider`, `model`, `operation` |
| `ractogateway_tokens_total` | Counter | `provider`, `model`, `token_type` |
| `ractogateway_cost_usd_total` | Counter | `provider`, `model` |
| `ractogateway_cache_hits_total` | Counter | `cache_type` |
| `ractogateway_cache_misses_total` | Counter | `cache_type` |
| `ractogateway_tool_calls_total` | Counter | `tool_name` |

### Methods

#### record_request

```python
def record_request(
    *,
    provider: str,
    model: str,
    operation: str,
    status: str,
    latency_s: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    tool_calls: list[ToolCallResult] | None = None,
) -> None
```

Record metrics for a completed LLM request.

**Parameters**

- **provider** — `"openai"`, `"google"`, or `"anthropic"`.
- **model** — Model identifier.
- **operation** — `"chat"`, `"stream"`, or `"embed"`.
- **status** — `"ok"` or `"error"`.
- **latency_s** — Request wall-clock latency **in seconds**.
- **input_tokens** — Prompt tokens consumed.
- **output_tokens** — Completion tokens produced.
- **tool_calls** — List of `ToolCallResult` objects from the response.

#### record_cache_hit

```python
def record_cache_hit(cache_type: str) -> None
```

Increment `ractogateway_cache_hits_total`. `cache_type` is `"exact"` or `"semantic"`.

#### record_cache_miss

```python
def record_cache_miss(cache_type: str) -> None
```

Increment `ractogateway_cache_misses_total`. `cache_type` is `"exact"` or `"semantic"`.

#### generate_latest

```python
def generate_latest(self) -> str
```

Return current metrics in Prometheus text exposition format (UTF-8 string).
Useful for testing without starting an HTTP server.

---

## PrometheusExporter

```python
class PrometheusExporter
```

HTTP server that exposes metrics at `/metrics` for Prometheus scraping.

### Constructor

```python
PrometheusExporter(
    port: int = 8000,
    registry: CollectorRegistry | None = None,
)
```

**Parameters**

- **port** — TCP port to listen on. Defaults to `8000`.
- **registry** — Custom registry. If `None`, uses the global `prometheus_client.REGISTRY`.

### Methods

#### start

```python
def start(self) -> None
```

Start the HTTP server in a background daemon thread. Idempotent.

#### stop

```python
def stop(self) -> None
```

Stop the HTTP server. Safe to call even if not started.

#### is_running (property)

```python
@property
def is_running(self) -> bool
```

`True` if the HTTP server thread is running.

---

## ModelPricing

```python
class ModelPricing(BaseModel)
```

USD cost per 1 million tokens for a specific model.

**Fields**

- **input_per_million** (`float`) — Price in USD for 1M input (prompt) tokens.
- **output_per_million** (`float`) — Price in USD for 1M output (completion) tokens.

---

## SpanRecord

```python
class SpanRecord(BaseModel)
```

In-memory span record.  Captured when `RactoTracer(in_memory=True)`.

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | OTEL span name (`"llm.chat"` or `"llm.embed"`) |
| `provider` | `str` | — | `"openai"` / `"google"` / `"anthropic"` |
| `model` | `str` | — | Model identifier |
| `operation` | `str` | — | `"chat"` / `"stream"` / `"embed"` |
| `latency_ms` | `float` | — | Wall-clock latency in milliseconds (≥ 0) |
| `input_tokens` | `int` | `0` | Prompt tokens |
| `output_tokens` | `int` | `0` | Completion tokens |
| `cost_usd` | `float` | `0.0` | Estimated USD cost |
| `cache_hit` | `str` | `"miss"` | `"exact"` / `"semantic"` / `"miss"` |
| `tool_calls` | `int` | `0` | Number of tool calls |
| `status` | `str` | `"ok"` | `"ok"` or `"error"` |
| `error_type` | `str \| None` | `None` | Exception class name |
| `timestamp` | `float` | `time.time()` | Unix timestamp of recording |

---

## DEFAULT_COST_TABLE

```python
DEFAULT_COST_TABLE: dict[str, ModelPricing]
```

Built-in pricing table with 40+ models.  Covers OpenAI (GPT-4o, GPT-4o-mini, o1, o3-mini,
…), Anthropic (Claude Opus/Sonnet/Haiku across generations), and Google (Gemini 2.0 Flash,
Gemini 2.5 Pro, Gemini 1.5, …).

---

## compute_cost

```python
def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float
```

Compute estimated USD cost from token counts using `DEFAULT_COST_TABLE`.
Returns `0.0` for unknown models.

**Parameters**

- **model** — Model identifier string.
- **input_tokens** — Number of input tokens.
- **output_tokens** — Number of output tokens.

**Returns** — Estimated cost in USD.
