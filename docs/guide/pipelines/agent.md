# AgentPipeline

`AgentPipeline` is an autonomous **ReAct (Reason + Act)** agent that solves
multi-step tasks by iteratively reasoning, calling tools, and observing
results.  It works with **any** RactoGateway developer kit (OpenAI, Google,
Anthropic, Ollama, HuggingFace) and requires no extra dependencies for the
core loop.

---

## How it works

```
┌────────────────────────────────────────────────────────┐
│  GOAL                                                  │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  LLM receives:                                   │  │
│  │    system prompt (tool list + rules)             │  │
│  │    transcript (goal + all prior steps)           │  │
│  │                                                  │  │
│  │  LLM outputs:                                    │  │
│  │    {"thought": "…", "tool_name": "…",            │  │
│  │     "tool_input": {…}}                           │  │
│  └──────────────┬───────────────────────────────────┘  │
│                 │                                       │
│         tool_name == "finish"? ──── Yes ───► RESULT    │
│                 │ No                                    │
│                 ▼                                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ToolExecutor.execute(tool_name, tool_input)     │  │
│  │  → observation (string, max 4 000 chars)         │  │
│  └──────────────┬───────────────────────────────────┘  │
│                 │                                       │
│         steps < max_steps? ──── No ────► MAX_STEPS     │
│                 │ Yes                                   │
│                 └──────────────► repeat                 │
└────────────────────────────────────────────────────────┘
```

The LLM **never sees raw function signatures** — every tool is described in
plain English inside the system prompt, making the pattern provider-agnostic.

---

## Installation

```bash
# Core agent — no extra deps
pip install ractogateway

# Optional: http_get tool (fetches URLs)
pip install ractogateway[pipelines-agent-http]
```

---

## Quickstart

```python
from ractogateway.openai_developer_kit import Chat
from ractogateway.pipelines.agent import AgentPipeline

# 1. Define your tools as plain Python functions
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    # In real code: call a weather API here
    return f"Sunny, 22 °C in {city}"

def convert_currency(amount: float, from_cur: str, to_cur: str) -> str:
    """Convert an amount between two currency codes (e.g. USD, EUR)."""
    rates = {"USD_EUR": 0.92, "EUR_USD": 1.09}
    rate = rates.get(f"{from_cur}_{to_cur}", 1.0)
    return f"{amount * rate:.2f} {to_cur}"

# 2. Build the agent
agent = AgentPipeline(
    kit=Chat(model="gpt-4o-mini"),
    tools=[get_weather, convert_currency],
    max_steps=8,
    safe_mode=True,
)

# 3. Run!
result = agent.run("What is the weather in Paris, and convert 100 USD to EUR?")

print(result.final_answer)
print(f"Steps taken: {result.usage.steps_taken}")
print(f"Tokens used: {result.usage.total_tokens}")
print(result.to_markdown())  # Full step trace
```

---

## Async usage (FastAPI, async servers)

```python
# Sync + async in the same class:
result = await agent.arun("What is 6 * 7?")

# Or use AsyncAgentPipeline for async-only servers:
from ractogateway.pipelines.agent import AsyncAgentPipeline

agent = AsyncAgentPipeline(
    kit=Chat(model="gpt-4o-mini"),
    tools=[get_weather],
)

# FastAPI endpoint:
@app.post("/ask")
async def ask(question: str) -> dict:
    result = await agent.run(question)
    return {"answer": result.final_answer, "steps": len(result.steps)}
```

---

## Constructor reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kit` | Any kit | — | RactoGateway developer kit |
| `tools` | `list[Callable]` | `None` | Plain Python callables to register |
| `rag_pipeline` | `RactoRAG` | `None` | Auto-registers a `rag_search` tool |
| `sql_pipeline` | `SQLAnalystPipeline` | `None` | Auto-registers a `sql_query` tool |
| `enable_http` | `bool` | `False` | Register `http_get` tool (needs httpx) |
| `agent_memory` | dict-like | `None` | Auto-registers `memory_read`/`memory_write` |
| `max_steps` | `int` | `10` | Hard cap on tool calls |
| `system_prompt` | `str` | `None` | Fully override the generated prompt |
| `extra_rules` | `str` | `""` | Append a custom rule line |
| `safe_mode` | `bool` | `False` | Return error instead of raising |
| `tracer` | `RactoTracer` | `None` | OpenTelemetry tracing |
| `metrics` | middleware | `None` | Prometheus metrics |
| `rate_limiter` | duck-typed | `None` | `check_and_consume(user_id, tokens)` |
| `user_id` | `str` | `"default"` | Default rate-limiter user |

`run()` and `arun()` accept per-call overrides: `max_steps`, `user_id`, `session_id`.

---

## Built-in tools

| Tool | Registered when | Description |
|------|-----------------|-------------|
| `finish` | Always | Signals task completion; sets `final_answer` |
| `rag_search` | `rag_pipeline=` provided | Searches a `RactoRAG` knowledge base |
| `sql_query` | `sql_pipeline=` provided | Natural-language SQL via `SQLAnalystPipeline` |
| `http_get` | `enable_http=True` | Fetches URL content (requires `httpx`) |
| `memory_read` | `agent_memory=` provided | Reads a key from agent memory |
| `memory_write` | `agent_memory=` provided | Writes a key to agent memory |

---

## Writing custom tools

Any Python callable becomes a tool.  The docstring's **first line** is shown to
the LLM as the tool's description, so make it descriptive:

```python
def search_products(query: str, max_results: int = 5) -> str:
    """Search the product catalogue and return matching item names and prices."""
    # … your implementation …
    return "\n".join(f"{p['name']}: ${p['price']}" for p in results[:max_results])
```

Async tools work too — `ToolExecutor` awaits them when `arun()` is used:

```python
async def get_stock_price(ticker: str) -> str:
    """Fetch the latest stock price for a ticker symbol (e.g. AAPL)."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://api.example.com/price/{ticker}")
        return r.json()["price"]
```

The `@tool` decorator from `ractogateway.tools` sets `__tool_name__` to
control the name exposed to the LLM:

```python
from ractogateway.tools import tool

@tool(name="lookup_customer")
def get_customer_by_id(customer_id: str) -> str:
    """Look up a customer record by their unique ID."""
    ...
```

---

## Plugging in RAG and SQL

```python
from ractogateway.rag import RactoRAG
from ractogateway.pipelines import SQLAnalystPipeline, AgentPipeline
from ractogateway.openai_developer_kit import Chat

kit = Chat(model="gpt-4o")

# RAG pipeline (documents already indexed)
rag = RactoRAG(...)

# SQL pipeline (connected to a database)
sql = SQLAnalystPipeline(
    kit=kit,
    connection_string="postgresql://user:pass@localhost/sales",
)

# Agent that can search docs AND query the database
agent = AgentPipeline(
    kit=kit,
    rag_pipeline=rag,
    sql_pipeline=sql,
    max_steps=12,
)

result = agent.run(
    "What were the top 3 products in Q1, and summarise what the docs say about them?"
)
print(result.final_answer)
```

---

## Memory persistence between turns

Use `agent_memory` for short-term key-value storage within a single session,
or plug in a Redis-backed store for multi-session persistence:

```python
from ractogateway.redis import RedisChatMemory   # for multi-session
from ractogateway.pipelines.agent import AgentPipeline

# Simple dict memory (in-process)
memory = {}

agent = AgentPipeline(
    kit=kit,
    tools=[get_weather],
    agent_memory=memory,
    max_steps=10,
)

# The agent can now call memory_write / memory_read between steps
result = agent.run("Remember that Alice prefers Celsius, then tell me her preferred unit.")
```

---

## Rate limiting

```python
from ractogateway.redis import RedisRateLimiter, RateLimitConfig

limiter = RedisRateLimiter(
    config=RateLimitConfig(max_tokens_per_minute=100),
    url="redis://localhost:6379",
)

agent = AgentPipeline(
    kit=kit,
    tools=[get_weather],
    rate_limiter=limiter,
    user_id="tenant_42",
)

# Per-call override:
result = agent.run("Task", user_id="tenant_99")
```

---

## Reading the result

```python
result = agent.run("What is the GDP of Germany?")

# Basic
print(result.final_answer)          # "The GDP of Germany is ~4.1 trillion USD."
print(result.succeeded())           # True
print(result.stop_reason)           # StopReason.FINISHED
print(result.usage.total_tokens)    # 842

# Steps
for step in result.steps:
    print(f"[{step.step_num}] {step.tool_name}: {step.thought}")

# Aggregated helpers
tool_calls = result.get_tool_calls()     # [(name, input), ...]
observations = result.get_observations() # ["obs1", "obs2", ...]

# Export
json_str = result.to_json()             # JSON string
result.to_json("run_result.json")       # write to file

md_str = result.to_markdown()           # Markdown trace string
result.to_markdown("run_trace.md")      # write to file
```

Sample Markdown output:

```markdown
# Agent Run

**Goal:** What is the weather in Paris?
**Status:** `finished` | **Steps:** 2 | **Tokens:** 312

---

## Step 1: `get_weather`

**Thought:** I need to call get_weather for Paris.

**Input:**
```json
{
  "city": "Paris"
}
```

**Observation:** (12 ms)
```
Sunny, 22 °C in Paris
```

---

## FINISH: `finish`

**Thought:** I have the weather data.

**Input:**
```json
{
  "answer": "It is sunny and 22 °C in Paris."
}
```

---

## Final Answer

It is sunny and 22 °C in Paris.
```

---

## Telemetry

```python
from ractogateway.telemetry import RactoTracer

tracer = RactoTracer(otlp_endpoint="http://localhost:4317")

agent = AgentPipeline(kit=kit, tools=[get_weather], tracer=tracer)
result = agent.run("Weather in Tokyo?")
```

Each step emits a span named `agent_step` with attributes:
`step_num`, `tool_name`, `duration_ms`, `input_tokens`, `output_tokens`.

---

## Error handling

| Scenario | `safe_mode=False` | `safe_mode=True` |
|----------|-------------------|------------------|
| LLM raises | exception propagates | `result.error` set, `stop_reason=ERROR` |
| Unknown tool | observation contains `ERROR: Unknown tool '...'` | same |
| Tool raises | observation contains `ERROR executing '...'` | same |
| Rate limited | `AgentRateLimitExceededError` raised | same (not caught) |
| Max steps hit | `result.stop_reason = MAX_STEPS` | same |

---

## Industrial use cases

| Industry | Example goal |
|----------|--------------|
| E-commerce | "Find the 3 cheapest in-stock laptops and compare specs" |
| Finance | "Summarise Q3 earnings from the docs and pull revenue from DB" |
| Healthcare | "Check patient record, retrieve drug interactions, and draft a note" |
| DevOps | "Check which services are down and fetch their last error logs" |
| Legal | "Search case law for relevant precedents on data-privacy breaches" |
| Customer support | "Identify the customer's issue, check their order, and draft a reply" |
