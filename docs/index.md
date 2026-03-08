# RactoGateway

**One Python package for production-grade AI development.**

RactoGateway is a unified AI SDK that gives you one clean interface for OpenAI,
Google Gemini, Anthropic Claude, Ollama (local), and HuggingFace. It combines
prompt engineering, strict Pydantic validation, tool calling, streaming,
embeddings, fine-tuning, RAG, and production infrastructure in one library.

## Why RactoGateway?

Every LLM provider has a different SDK, request format, response structure, and
tool-calling schema. Production AI systems often turn into glue code and brittle
parsers.

RactoGateway solves this by providing:

- `RactoPrompt` (RACTO) for structured prompting with anti-hallucination guardrails.
- Five unified developer kits: OpenAI (`gpt`), Google (`gemini`), Anthropic (`claude`), Ollama (`local`), HuggingFace (`hf`).
- Strict typed models for input/output and robust response validation.
- Unified tool calling via `ToolRegistry`.
- Typed streaming chunks and async support across providers.
- End-to-end retrieval with `RactoRAG` plus vectorless `PageIndexRAG`.
- Turn-key workflows: `SQLAnalystPipeline`, `ListClassifierPipeline`, `VideoProcessorPipeline`, `AgentPipeline`.
- Production controls: exact cache, semantic cache, routing, truncation, batch.
- Ops modules for Redis, Celery, Kafka, MCP, and telemetry.

### Use-Case Map

| Use case | Typical friction | How RactoGateway helps |
| --- | --- | --- |
| Build chat/API assistants | Provider SDK drift and response shape mismatch | One `ChatConfig` + one `LLMResponse` model across providers |
| Return strict JSON for automation | Markdown fenced JSON and schema drift | `RactoPrompt(output_format=YourModel)` embeds schema and enforces shape |
| Add tools into workflows | Different function-calling formats per vendor | Register Python tools once with `ToolRegistry` |
| Build RAG assistants | Stitching readers/chunkers/embedders/stores manually | `RactoRAG` handles ingest -> retrieve -> generate |
| Keep costs predictable | Duplicate calls and oversized model usage | Cache + routing + truncation + batch controls |
| Operate on multiple servers | In-memory cache/memory does not scale | Redis modules for distributed cache, memory, and rate limits |
| Run long jobs safely | Request-thread failures and retries | `RactoCeleryWorker` for retries and background execution |

### Why It Stands Different

| Dimension | Typical approach | RactoGateway approach | Practical impact |
| --- | --- | --- | --- |
| Provider support | Rebuild when switching SDK | Same mental model across providers | Faster migration and multi-provider strategy |
| Prompt reliability | Ad-hoc prompt strings | Structured RACTO prompts | More consistent outputs |
| Output safety | Manual `json.loads` parsing | Typed validation + normalized responses | Fewer runtime failures |
| Tool integration | Vendor-specific tool payloads | Single `ToolRegistry` abstraction | Less integration code |
| RAG delivery | Many separate libraries | One orchestrator with pluggable parts | Faster production rollout |
| Scale and operations | Infra bolted on later | Redis/Celery/Kafka/MCP first-class modules | Better reliability and throughput |

### Platform Architecture

RactoGateway is designed as one composable stack rather than disconnected helper utilities:

| Layer | Core modules | What you get |
| --- | --- | --- |
| Prompt and output control | `RactoPrompt`, `RactoFile` | Structured prompts (RACTO), anti-hallucination guardrails, deterministic output shape |
| Multi-provider chat | `openai_developer_kit`, `google_developer_kit`, `anthropic_developer_kit`, `ollama_developer_kit`, `huggingface_developer_kit` | One mental model across cloud and local LLM providers |
| Tool execution | `ToolRegistry`, `tool` decorator | Define Python tools once and execute them through a provider-agnostic interface |
| Structured response safety | `response_model` support + strict validation | Typed results instead of brittle raw JSON parsing |
| Retrieval pipeline | `RactoRAG`, `PageIndexRAG`, readers/chunkers/embedders/stores | Ingest -> retrieve -> generate for document-grounded answers |
| Turn-key workflows | `SQLAnalystPipeline`, `ListClassifierPipeline`, `VideoProcessorPipeline`, `AgentPipeline` | Complete domain workflows with sync and async variants |
| Cost and performance controls | exact cache, semantic cache, routing, truncation, batch | Lower spend, lower latency, and better throughput |
| Production operations | Redis, Celery, Kafka, MCP, telemetry | Distributed memory/cache/rate-limits, background jobs, streaming, and observability |

## End-to-End Pipeline in Practice

Use the library as a composable delivery pipeline instead of isolated API calls:

1. Define behavior with `RactoPrompt` (role, aim, constraints, tone, output).
2. Choose any provider kit (`gpt`, `gemini`, `claude`, `local`, or `hf`).
3. Call `chat()` / `stream()` / `embed()` with typed config models.
4. Optionally attach tools via `ToolRegistry` for function execution.
5. Optionally add retrieval with `RactoRAG` or `PageIndexRAG`.
6. Optionally move to prebuilt pipelines for SQL analytics, classification, video intelligence, or agentic loops.
7. Add production controls (cache, routing, truncation, batch, Redis, Celery).
8. Observe and operate with telemetry, Kafka integration, and MCP interoperability.

## Pipeline Catalog

| Pipeline | Input | Output | Typical use case |
| --- | --- | --- | --- |
| `SQLAnalystPipeline` | Natural language question + DB connection | SQL, result tables, narrative answer, optional chart | BI copilots, operations reporting, analytics assistants |
| `ListClassifierPipeline` | User text + controlled options list | Single/multi label, confidence, optional reasoning | Ticket routing, intent detection, workflow triage |
| `VideoProcessorPipeline` | Video path/URL/YouTube/bytes | Transcript, frame analysis, section summaries, optional RAG storage | Lecture indexing, training content QA, media intelligence |
| `AgentPipeline` | Goal + tools | Multi-step tool traces + final answer | ReAct-style automation, tool-driven agents, research workflows |

## Real-World Use Cases (Implementation Blueprints)

The examples below show how teams use RactoGateway as a full delivery stack,
not just a chat wrapper.

| Scenario | Primary modules | What ships to production |
| --- | --- | --- |
| Customer support copilot | `ListClassifierPipeline`, `ToolRegistry`, `RactoRAG`, Redis modules | Auto-routing, grounded answers, strict response schema, low-latency cached replies |
| BI and data analyst assistant | `SQLAnalystPipeline`, `RactoPrompt`, typed models | Natural language to SQL, safe query execution, markdown answer, optional charts |
| Internal knowledge assistant | `RactoRAG` or `PageIndexRAG`, `RactoPrompt` | Policy and SOP answers grounded on private docs with source-aware retrieval |
| Video intelligence pipeline | `VideoProcessorPipeline`, optional RAG store | Transcript + frame analysis + summary, then searchable knowledge base from videos |
| Agentic back-office automation | `AgentPipeline`, `ToolRegistry`, `MCP`, Celery | Multi-step tool execution with bounded steps, retries, and background orchestration |

### 1) Customer Support Copilot (SaaS)

**Business goal:** Reduce first-response time while keeping answers accurate and auditable.

**Typical stack:**

1. Route incoming tickets with `ListClassifierPipeline`.
2. Use `ToolRegistry` for account lookup, billing state, and CRM actions.
3. Ground responses with `RactoRAG` over help-center and policy docs.
4. Enforce structured outputs with `response_model` (no free-form drift).
5. Run Redis cache + memory + rate limit for multi-server deployments.

```python
from pydantic import BaseModel
from ractogateway import openai_developer_kit as gpt
from ractogateway.pipelines import ListClassifierPipeline
from ractogateway.redis import RedisExactCache, RedisChatMemory

class SupportReply(BaseModel):
    route: str
    reply: str
    escalate: bool

classifier = ListClassifierPipeline(
    kit=gpt.Chat(model="gpt-4o-mini"),
    options=["Billing", "Technical Support", "Account", "Sales"],
)
route = classifier.run("My invoice is wrong and payment failed").first

kit = gpt.Chat(
    model="gpt-4o",
    exact_cache=RedisExactCache(url="redis://localhost:6379/0", ttl_seconds=3600),
)
result = kit.chat(
    gpt.ChatConfig(
        user_message="Resolve this ticket with account-safe steps.",
        response_model=SupportReply,
    )
)
```

### 2) BI Analyst Copilot

**Business goal:** Let business teams ask plain-English data questions and get reliable answers.

**Typical stack:**

1. `SQLAnalystPipeline` for NL -> SQL -> execution -> analysis.
2. Read-only guardrails + `safe_mode=True` for operational safety.
3. Optional chart generation for dashboard-ready output.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.pipelines import SQLAnalystPipeline

pipeline = SQLAnalystPipeline(kit=gpt.Chat(model="gpt-4o"), safe_mode=True)
result = pipeline.run(
    user_query="Top 10 products by revenue growth this quarter",
    connection_string="postgresql://user:pass@localhost:5432/warehouse",
)
print(result.sql_query)
print(result.answer)
```

### 3) Internal Knowledge Assistant (Policies, SOPs, Engineering Docs)

**Business goal:** Replace document hunt with grounded Q&A over private content.

**Typical stack:**

1. Ingest docs with `RactoRAG` (`pdf`, `docx`, `xlsx`, html, text).
2. Pick embedder + vector store based on your environment.
3. Use a strict prompt and retrieval filters for domain-safe answers.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.rag import RactoRAG
from ractogateway.rag.embedders import OpenAIEmbedder
from ractogateway.rag.stores import ChromaStore

rag = RactoRAG(
    vector_store=ChromaStore(collection="internal_docs", persist_directory="./db"),
    embedder=OpenAIEmbedder(model="text-embedding-3-large"),
    llm_kit=gpt.Chat(model="gpt-4o"),
)
rag.ingest_dir("./knowledge_base", pattern="**/*")
response = rag.query("What is our production incident escalation policy?", top_k=5)
print(response.answer)
```

### 4) Video Learning Intelligence

**Business goal:** Convert training recordings into searchable, reusable knowledge.

**Typical stack:**

1. `VideoProcessorPipeline` for frame dedup + transcription + visual analysis.
2. Generate sectioned summaries for quick comprehension.
3. Store outputs in RAG for downstream question answering.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.pipelines import VideoProcessorPipeline, TranscriberBackend

video = VideoProcessorPipeline(
    kit=gpt.Chat(model="gpt-4o"),
    transcriber=TranscriberBackend.FASTER_WHISPER,
    generate_summary=True,
    safe_mode=True,
)
report = video.run("onboarding_session.mp4")
print(report.summary)
```

### 5) Agentic Operations Automation

**Business goal:** Automate multi-step tasks (fetch data, reason, call tools, return final action plan).

**Typical stack:**

1. `AgentPipeline` with approved tool set (`SQL`, HTTP, RAG, custom Python tools).
2. `max_steps` and `safe_mode` for bounded execution.
3. Optional Celery for background and retry-safe execution.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.pipelines import AgentPipeline

def get_inventory(sku: str) -> str:
    return f"SKU {sku}: 42 units in stock"

agent = AgentPipeline(
    kit=gpt.Chat(model="gpt-4o-mini"),
    tools=[get_inventory],
    max_steps=6,
    safe_mode=True,
)
result = agent.run("Check stock for SKU-4481 and suggest reorder action.")
print(result.final_answer)
```

### 6) Cost-Controlled Multi-Provider Delivery

**Business goal:** Keep quality high while controlling spend and avoiding vendor lock-in.

**Typical stack:**

1. Start with one prompt contract (`RactoPrompt`).
2. Switch provider kits without changing business logic.
3. Add `ExactMatchCache`, `SemanticCache`, and routing for cost-performance balance.
4. Use batch APIs for large offline workloads.

Result: one codebase, provider flexibility, and predictable cost envelopes as usage scales.

## Documentation Paths

- New to the library: start with [Installation](installation.md) and [Quick Start](quickstart.md).
- Building assistants and APIs: see [Developer Kits](guide/developer_kits.md), [Prompt Engine](guide/prompt_engine.md), and [Tools](guide/tools.md).
- Building retrieval systems: see [RAG](guide/rag.md), [Embeddings](guide/embeddings.md), and [Pipelines](guide/pipelines.md).
- Running in production: see [Cache](guide/cache.md), [Routing](guide/routing.md), [Redis](guide/redis.md), [Celery](guide/celery.md), [Kafka](guide/kafka.md), and [MCP](guide/mcp.md).
- Improving LLM discoverability: see [LLM Discovery Guide](guide/llm_discovery.md) and root files `llms.txt`, `llms-full.txt`, and `robots.txt`.

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide/userguide
guide/llm_discovery
guide/prompt_engine
guide/developer_kits
guide/ollama
guide/huggingface
guide/streaming
guide/tools
guide/embeddings
guide/chain_of_thought
guide/native_thinking
guide/finetune
guide/rag
guide/pipelines
guide/batch
guide/cache
guide/routing
guide/truncation
guide/mcp
guide/redis
guide/celery
guide/kafka
```

```{toctree}
:maxdepth: 3
:caption: API Reference

api/index
```
