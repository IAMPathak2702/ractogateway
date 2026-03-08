# RactoGateway

**One Python package for production-grade AI development.**

RactoGateway is a unified AI SDK that gives you one clean interface for OpenAI,
Google Gemini, Anthropic Claude, Ollama (local), and HuggingFace. It combines
prompt engineering, strict Pydantic validation, tool calling, streaming,
embeddings, fine-tuning, RAG, and production infrastructure in one library.

## Why Teams Choose RactoGateway

Most AI projects become hard to maintain because they combine many disconnected
pieces: one SDK for chat, another for RAG, custom parsers for JSON, custom logic
for tools, and separate systems for reliability and scaling.

RactoGateway packages those layers into a single architecture:

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

## Documentation Paths

- New to the library: start with [Installation](installation.md) and [Quick Start](quickstart.md).
- Building assistants and APIs: see [Developer Kits](guide/developer_kits.md), [Prompt Engine](guide/prompt_engine.md), and [Tools](guide/tools.md).
- Building retrieval systems: see [RAG](guide/rag.md), [Embeddings](guide/embeddings.md), and [Pipelines](guide/pipelines.md).
- Running in production: see [Cache](guide/cache.md), [Routing](guide/routing.md), [Redis](guide/redis.md), [Celery](guide/celery.md), [Kafka](guide/kafka.md), and [MCP](guide/mcp.md).

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
