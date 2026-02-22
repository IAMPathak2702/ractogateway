# RactoGateway

**One Python package for all production-grade LLM solutions.**

RactoGateway is a unified AI SDK that gives you a single, clean interface to OpenAI, Google Gemini, and Anthropic Claude — with built-in anti-hallucination prompting, strict Pydantic validation, streaming, tool calling, embeddings, fine-tuning, and a full RAG pipeline. No more messy JSON dicts. No more provider lock-in. No more inconsistent response formats.

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.3-blue.svg)](https://pypi.org/project/ractogateway/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-GitHub-green.svg)](https://github.com/IAMPathak2702/RactoGateway)

---

## Table of Contents

- [Why RactoGateway?](#why-ractogateway)
- [Installation](#installation)
- [5-Line Quick Start](#5-line-quick-start)
- [RACTO Prompt Engine](#racto-prompt-engine)
- [Developer Kits](#developer-kits)
- [Streaming](#streaming)
- [Async Support](#async-support)
- [Embeddings](#embeddings)
- [Tool Calling](#tool-calling)
- [Validated Response Models](#validated-response-models)
- [Multi-turn Conversations](#multi-turn-conversations)
- [Multimodal Attachments — Images & Files](#multimodal-attachments)
- [Low-Level Gateway](#low-level-gateway)
- [Switching Providers](#switching-providers)
- [Fine-Tuning](#fine-tuning)
- [MCP (Model Context Protocol)](#mcp-model-context-protocol)
- [RAG — Retrieval-Augmented Generation](#rag)
- [Performance & Cost Optimization](#performance--cost-optimization)
  - [Exact-Match Cache](#exact-match-cache)
  - [Semantic Cache](#semantic-cache)
  - [Cost-Aware Routing](#cost-aware-routing)
  - [Token Truncation](#token-truncation)
  - [Batch Processing](#batch-processing)
  - [Combining All Optimizations](#combining-all-optimizations)
- [Redis Infrastructure](#redis-infrastructure)
  - [RedisExactCache — Distributed Response Cache](#redisexactcache--distributed-response-cache)
  - [RedisRateLimiter — Fleet-Wide Rate Limiting](#redisratelimiter--fleet-wide-rate-limiting)
  - [RedisChatMemory — Sliding-Window Conversation History](#redischatmemory--sliding-window-conversation-history)
  - [Production Pattern — Combining Redis Utilities](#production-pattern--combining-redis-utilities)
- [Celery Task Queue](#celery-task-queue)
  - [Never-Fail LLM Generation](#never-fail-llm-generation)
  - [Background Document Ingestion](#background-document-ingestion)
  - [Parallel Batch Inference](#parallel-batch-inference)
  - [RetryConfig — Exponential Backoff Policy](#retryconfig--exponential-backoff-policy)
- [Environment Variables](#environment-variables)

---

## Why RactoGateway?

Every LLM provider has a different SDK, different request format, different response structure, and different tool-calling schema. Building production AI applications means writing glue code, parsing deeply nested objects, and manually stripping markdown fences from JSON responses.

RactoGateway solves this by providing:

- **RACTO Prompt Engine** — a structured prompt framework (Role, Aim, Constraints, Tone, Output) that compiles into optimized, anti-hallucination system prompts
- **Three Developer Kits** — `gpt` (OpenAI), `gemini` (Google), `claude` (Anthropic) — each with `chat()`, `achat()`, `stream()`, `astream()`, `embed()`, and `aembed()`
- **Strict Pydantic models** for every input and output — no raw dicts anywhere
- **Automatic JSON parsing** — responses are cleaned of markdown fences and auto-parsed
- **Unified tool calling** — define tools once as Python functions, use them with any provider
- **Streaming with typed chunks** — every `StreamChunk` has `.delta.text`, `.accumulated_text`, `.is_final`, `.usage`
- **RAG pipeline** — ingest files, embed, store, retrieve, and generate answers with one class
- **Low-level Gateway** — wraps any adapter for direct prompt execution without `ChatConfig`
- **Exact-match cache** — SHA-256 LRU cache eliminates duplicate API calls with zero latency
- **Semantic cache** — cosine-similarity cache returns cached answers for semantically equivalent queries
- **Cost-aware routing** — `model="auto"` dynamically picks the cheapest model that can handle the request
- **Token truncation** — automatically trims conversation history before hitting context limits
- **Batch processing** — submit thousands of tasks at ~50 % cost via OpenAI & Anthropic Batch APIs
- **Redis distributed cache** — drop-in `RedisExactCache` shares the response cache across all servers in a fleet
- **Redis rate limiter** — fleet-wide token-budget enforcement per user ID, safe across concurrent processes
- **Redis chat memory** — sliding-window conversation history backed by Redis Lists, survives rolling deployments
- **Celery task queue** — background generation, retry-safe workflows, and parallel inference across worker nodes

---

## Installation

```bash
# Core package (includes RACTO prompt engine and tool registry)
pip install ractogateway

# With a specific LLM provider
pip install ractogateway[openai]
pip install ractogateway[google]
pip install ractogateway[anthropic]

# All LLM providers
pip install ractogateway[all]

# RAG: base readers + NLP processing
pip install ractogateway[rag]

# RAG: everything (all readers, stores, embedders)
pip install ractogateway[rag-all]

# RAG: individual extras
pip install ractogateway[rag-pdf]       # PDF support
pip install ractogateway[rag-word]      # .docx support
pip install ractogateway[rag-excel]     # .xlsx support
pip install ractogateway[rag-image]     # image OCR support
pip install ractogateway[rag-nlp]       # lemmatizer NLP processing

# RAG: vector stores
pip install ractogateway[rag-chroma]    # ChromaDB
pip install ractogateway[rag-faiss]     # FAISS
pip install ractogateway[rag-pinecone]  # Pinecone
pip install ractogateway[rag-qdrant]    # Qdrant
pip install ractogateway[rag-weaviate]  # Weaviate
pip install ractogateway[rag-milvus]    # Milvus
pip install ractogateway[rag-pgvector]  # PostgreSQL pgvector

# RAG: embedding providers
pip install ractogateway[rag-voyage]    # Voyage AI embeddings

# MCP extras
pip install ractogateway[mcp]           # MCP core (stdio + SSE client)
pip install ractogateway[mcp-sse]       # MCP SSE server (Starlette + Uvicorn)

# Performance extras
pip install ractogateway[cache]         # tiktoken for precise token counting

# Redis infrastructure (distributed cache, rate limiter, chat memory)
pip install ractogateway[redis]

# Celery task queue (background jobs + retries + parallel fan-out)
pip install ractogateway[celery]

# Development (all providers + testing + linting)
pip install ractogateway[dev]
```

**Requirements:** Python 3.10+, Pydantic 2.0+

---

## 5-Line Quick Start

This is the absolute minimum to get a response from any AI — no configuration needed beyond your API key:

```python
from ractogateway import openai_developer_kit as gpt, RactoPrompt

# 1. Describe what you want the AI to do
prompt = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Answer the user's question clearly.",
    constraints=["Be concise."],
    tone="Friendly",
    output_format="text",
)

# 2. Create your AI chat (reads OPENAI_API_KEY from environment automatically)
kit = gpt.Chat(model="gpt-4o", default_prompt=prompt)

# 3. Ask something!
response = kit.chat(gpt.ChatConfig(user_message="What is Python?"))
print(response.content)
# "Python is a beginner-friendly, high-level programming language used for web
#  development, data science, AI, automation, and much more."
```

That's it. Swap `gpt` for `gemini` or `claude` and the exact same code works with Google or Anthropic.

---

## RACTO Prompt Engine

The **RACTO** principle structures every prompt into five unambiguous sections so the model always knows exactly what to do — and what NOT to do.

| Letter | Field | Purpose |
| :---: | --- | --- |
| **R** | `role` | Who the model is |
| **A** | `aim` | What it must accomplish |
| **C** | `constraints` | Hard rules it must never violate |
| **T** | `tone` | Communication style |
| **O** | `output_format` | Exact shape of the response |

### Defining a Prompt

```python
from ractogateway import RactoPrompt

prompt = RactoPrompt(
    role="You are a senior Python code reviewer at a Fortune 500 company.",
    aim="Review the given code for bugs, security vulnerabilities, and PEP-8 violations.",
    constraints=[
        "Only report issues you are certain about.",
        "Do not suggest stylistic preferences.",
        "If no issues are found, say so explicitly.",
        "Never fabricate code examples that you cannot verify.",
    ],
    tone="Professional and concise",
    output_format="json",
)
```

### All `RactoPrompt` Fields

| Field | Type | Required | Default | Description |
| --- | --- | :---: | --- | --- |
| `role` | `str` | Yes | — | Who the model is |
| `aim` | `str` | Yes | — | Task objective |
| `constraints` | `list[str]` | Yes | — | Hard rules (min 1 item) |
| `tone` | `str` | Yes | — | Communication style |
| `output_format` | `str \| type[BaseModel]` | Yes | — | `"json"`, `"text"`, `"markdown"`, free-form description, or a Pydantic class |
| `context` | `str \| None` | No | `None` | Domain background injected between AIM and CONSTRAINTS |
| `examples` | `list[dict] \| None` | No | `None` | Few-shot pairs — each dict requires `"input"` and `"output"` keys |
| `anti_hallucination` | `bool` | No | `True` | Append `[GUARDRAILS]` block |

### `RactoPrompt` Methods

| Method | Signature | Returns | Description |
| --- | --- | --- | --- |
| `compile()` | `() -> str` | `str` | Generate the full system prompt string |
| `__str__()` | `() -> str` | `str` | Shortcut for `compile()` |
| `to_messages()` | `(user_message, attachments=None, provider="generic") -> list[dict]` | `list[dict]` | Build a provider-ready message list |

### What `prompt.compile()` Produces

Calling `prompt.compile()` (or just `str(prompt)`) gives you the full system prompt:

```text
[ROLE]
You are a senior Python code reviewer at a Fortune 500 company.

[AIM]
Review the given code for bugs, security vulnerabilities, and PEP-8 violations.

[CONSTRAINTS]
- Only report issues you are certain about.
- Do not suggest stylistic preferences.
- If no issues are found, say so explicitly.
- Never fabricate code examples that you cannot verify.

[TONE]
Professional and concise

[OUTPUT]
Respond ONLY with valid JSON. Do NOT wrap the response in markdown code
fences (```json … ```) or add any commentary before or after the JSON object.

[GUARDRAILS]
- If you are unsure or lack sufficient information, state it explicitly rather than guessing.
- Do NOT fabricate facts, citations, URLs, statistics, or code that you cannot verify.
- Stick strictly to what is asked. Do not add unrequested information.
- If the answer requires assumptions, list each assumption explicitly before proceeding.
```

### Pydantic Model as Output Format

Pass a Pydantic model class as `output_format` and the full JSON Schema is embedded in the compiled prompt automatically:

```python
from pydantic import BaseModel

class CodeReview(BaseModel):
    issues: list[str]
    severity: str   # "low", "medium", "high"
    suggestion: str

prompt = RactoPrompt(
    role="You are a code reviewer.",
    aim="Review the code.",
    constraints=["Only report real issues."],
    tone="Concise",
    output_format=CodeReview,   # ← JSON Schema auto-embedded in prompt
)

print(prompt.compile())
```

Compiled output (OUTPUT section):

```text
[OUTPUT]
Respond ONLY with valid JSON that conforms exactly to the following JSON Schema.
Do NOT wrap the JSON in markdown code fences or add any text before or after it.

JSON Schema:
{
  "type": "object",
  "properties": {
    "issues": {"type": "array", "items": {"type": "string"}},
    "severity": {"type": "string"},
    "suggestion": {"type": "string"}
  },
  "required": ["issues", "severity", "suggestion"]
}
```

### Few-Shot Examples

```python
prompt = RactoPrompt(
    role="You are a sentiment classifier.",
    aim="Classify the sentiment of the user's text.",
    constraints=["Only output: positive, negative, or neutral."],
    tone="Concise",
    output_format="json",
    examples=[
        {"input": "I love this product!", "output": '{"sentiment": "positive"}'},
        {"input": "This is broken and useless.", "output": '{"sentiment": "negative"}'},
        {"input": "It arrived yesterday.", "output": '{"sentiment": "neutral"}'},
    ],
)
```

### `to_messages()` — Ready-to-Send Message List

**Input parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `user_message` | `str` | — | The end-user's query (required) |
| `attachments` | `list[RactoFile] \| None` | `None` | Optional file/image attachments |
| `provider` | `str` | `"generic"` | `"openai"`, `"anthropic"`, `"google"`, or `"generic"` |

**Output:** `list[dict[str, Any]]` — a list of message dicts ready to send to the provider

```python
messages = prompt.to_messages(
    "Review this: def add(a, b): return a + b",
    provider="openai",   # "openai" | "anthropic" | "google" | "generic"
)

# Output:
# [
#   {"role": "system", "content": "<compiled RACTO system prompt>"},
#   {"role": "user",   "content": "Review this: def add(a, b): return a + b"}
# ]
```

---

## Developer Kits

RactoGateway has three kits — one for each AI provider. Import them with names you already know, then call `.Chat(...)` to create your AI:

```python
from ractogateway import openai_developer_kit as gpt      # ChatGPT / OpenAI
from ractogateway import google_developer_kit as gemini   # Google Gemini
from ractogateway import anthropic_developer_kit as claude # Anthropic Claude
```

> **Note:** `and` is a reserved Python keyword in Python, so we use `claude` instead — cleaner anyway!

### Creating a Chat

Every kit exposes a `Chat` class — short, readable, and always works the same way:

```python
# Just pick your provider and model — that's it!
kit = gpt.Chat(model="gpt-4o")
kit = gemini.Chat(model="gemini-2.0-flash")
kit = claude.Chat(model="claude-sonnet-4-6")
```

The API key is read automatically from your environment variable (`OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`). No extra setup needed.

**Full constructor options (all optional except `model`):**

```python
# OpenAI / ChatGPT
kit = gpt.Chat(
    model="gpt-4o",                            # which model to use
    api_key="sk-...",                          # skip if OPENAI_API_KEY is set
    base_url="https://custom-proxy.com/v1",    # optional: Azure or custom proxy
    embedding_model="text-embedding-3-small",  # for embed() calls
    default_prompt=prompt,                     # auto-used in every chat if set
)

# Google Gemini
kit = gemini.Chat(
    model="gemini-2.0-flash",                  # which model to use
    api_key="AIza...",                         # skip if GEMINI_API_KEY is set
    embedding_model="text-embedding-004",      # for embed() calls
    default_prompt=prompt,                     # auto-used in every chat if set
)

# Anthropic Claude
kit = claude.Chat(
    model="claude-sonnet-4-6",                 # which model to use
    api_key="sk-ant-...",                      # skip if ANTHROPIC_API_KEY is set
    default_prompt=prompt,                     # auto-used in every chat if set
)
```

**`OpenAIDeveloperKit` / `gpt.Chat` constructor parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `str` | `"gpt-4o"` | Chat model identifier |
| `api_key` | `str \| None` | `None` | Falls back to `OPENAI_API_KEY` env var |
| `base_url` | `str \| None` | `None` | Azure OpenAI or proxy base URL |
| `embedding_model` | `str` | `"text-embedding-3-small"` | Default model for `embed()` calls |
| `default_prompt` | `RactoPrompt \| None` | `None` | Auto-used when `ChatConfig.prompt` is `None` |

**`GoogleDeveloperKit` / `gemini.Chat` constructor parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `str` | `"gemini-2.0-flash"` | Chat model identifier |
| `api_key` | `str \| None` | `None` | Falls back to `GEMINI_API_KEY` env var |
| `embedding_model` | `str` | `"text-embedding-004"` | Default model for `embed()` calls |
| `default_prompt` | `RactoPrompt \| None` | `None` | Auto-used when `ChatConfig.prompt` is `None` |

**`AnthropicDeveloperKit` / `claude.Chat` constructor parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `str` | — | Chat model identifier (required) |
| `api_key` | `str \| None` | `None` | Falls back to `ANTHROPIC_API_KEY` env var |
| `default_prompt` | `RactoPrompt \| None` | `None` | Auto-used when `ChatConfig.prompt` is `None` |

### Method Reference

| Method | `gpt` | `gemini` | `claude` | Input | Output |
| --- | :---: | :---: | :---: | --- | --- |
| `chat(config)` | Yes | Yes | Yes | `ChatConfig` | `LLMResponse` |
| `achat(config)` | Yes | Yes | Yes | `ChatConfig` | `LLMResponse` |
| `stream(config)` | Yes | Yes | Yes | `ChatConfig` | `Iterator[StreamChunk]` |
| `astream(config)` | Yes | Yes | Yes | `ChatConfig` | `AsyncIterator[StreamChunk]` |
| `embed(config)` | Yes | Yes | — | `EmbeddingConfig` | `EmbeddingResponse` |
| `aembed(config)` | Yes | Yes | — | `EmbeddingConfig` | `EmbeddingResponse` |

> Anthropic does not offer a native embedding API. Use the OpenAI or Google kit for embeddings.

---

### `ChatConfig` — Input Model

The single input object for `chat()`, `achat()`, `stream()`, and `astream()`.

```python
config = gpt.ChatConfig(
    user_message="Explain monads in simple terms.",   # required
    prompt=prompt,                                     # optional — overrides kit default
    temperature=0.3,                                   # 0.0–2.0, default 0.0
    max_tokens=2048,                                   # default 4096
    tools=my_tool_registry,                            # optional ToolRegistry
    response_model=MyPydanticModel,                    # optional output validation
    history=[                                          # optional multi-turn context
        gpt.Message(role=gpt.MessageRole.USER, content="What is FP?"),
        gpt.Message(role=gpt.MessageRole.ASSISTANT, content="Functional programming is..."),
    ],
    extra={"top_p": 0.9, "seed": 42},                 # provider-specific pass-through
)
```

**`ChatConfig` field reference:**

| Field | Type | Required | Default | Description |
| --- | --- | :---: | --- | --- |
| `user_message` | `str` | Yes | — | End-user's query (min 1 character) |
| `prompt` | `RactoPrompt \| None` | No | `None` | Overrides the kit's `default_prompt` for this call |
| `temperature` | `float` | No | `0.0` | Sampling temperature (0.0–2.0) |
| `max_tokens` | `int` | No | `4096` | Maximum tokens in the completion (>0) |
| `tools` | `ToolRegistry \| None` | No | `None` | Tool registry for function/tool calling |
| `response_model` | `type[BaseModel] \| None` | No | `None` | Validate JSON output against this Pydantic model |
| `history` | `list[Message]` | No | `[]` | Prior conversation turns for multi-turn chat |
| `extra` | `dict[str, Any]` | No | `{}` | Provider-specific pass-through kwargs (e.g. `top_p`, `seed`, `stop`) |

> **Note:** Either `ChatConfig.prompt` or the kit's `default_prompt` must be set — at least one is required.

---

### `Message` and `MessageRole`

Used to build conversation history for multi-turn chat.

```python
from ractogateway import openai_developer_kit as gpt

msg = gpt.Message(role=gpt.MessageRole.USER, content="What is Python?")
```

**`Message` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `role` | `MessageRole` | `SYSTEM`, `USER`, or `ASSISTANT` |
| `content` | `str` | The message text |

**`MessageRole` enum values:**

| Value | String | Description |
| --- | --- | --- |
| `MessageRole.SYSTEM` | `"system"` | System instruction |
| `MessageRole.USER` | `"user"` | Human turn |
| `MessageRole.ASSISTANT` | `"assistant"` | Model turn |

---

### `LLMResponse` — Chat Output

Returned by `chat()` and `achat()`. Same shape for all three providers.

```python
response = kit.chat(gpt.ChatConfig(user_message="What is 2 + 2?"))

response.content        # "4"  — cleaned text (markdown fences auto-stripped)
response.parsed         # None  (not JSON) or dict/list if JSON
response.tool_calls     # []   — list[ToolCallResult]
response.finish_reason  # FinishReason.STOP
response.usage          # {"prompt_tokens": 42, "completion_tokens": 5, "total_tokens": 47}
response.raw            # the unmodified provider response object (escape hatch)
```

**Full output example — JSON response:**

```python
prompt = RactoPrompt(
    role="You are a data extractor.",
    aim="Extract the person's name and age from the text.",
    constraints=["Return only JSON."],
    tone="Concise",
    output_format="json",
)
kit = gpt.Chat(model="gpt-4o", default_prompt=prompt)
response = kit.chat(gpt.ChatConfig(user_message="My name is Alice and I am 30 years old."))

print(response.content)
# '{"name": "Alice", "age": 30}'

print(response.parsed)
# {"name": "Alice", "age": 30}   ← auto-parsed Python dict, no json.loads() needed

print(response.finish_reason)
# FinishReason.STOP

print(response.usage)
# {"prompt_tokens": 78, "completion_tokens": 12, "total_tokens": 90}
```

**`LLMResponse` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `content` | `str \| None` | Cleaned text (markdown fences stripped) |
| `parsed` | `dict \| list \| None` | Auto-parsed JSON — `None` when response is not JSON |
| `tool_calls` | `list[ToolCallResult]` | Tool calls requested by the model |
| `finish_reason` | `FinishReason` | `STOP`, `TOOL_CALL`, `LENGTH`, `CONTENT_FILTER`, `ERROR` |
| `usage` | `dict[str, int]` | `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `raw` | `Any` | The unmodified provider response (escape hatch for advanced use) |

**`FinishReason` enum values:**

| Value | String | When set |
| --- | --- | --- |
| `FinishReason.STOP` | `"stop"` | Normal completion |
| `FinishReason.TOOL_CALL` | `"tool_call"` | Model requested a function/tool call |
| `FinishReason.LENGTH` | `"length"` | Hit `max_tokens` limit |
| `FinishReason.CONTENT_FILTER` | `"content_filter"` | Filtered by safety system |
| `FinishReason.ERROR` | `"error"` | Internal error |

---

## Streaming

`stream()` and `astream()` yield `StreamChunk` objects — one per streaming event.

```python
from ractogateway import openai_developer_kit as gpt, RactoPrompt

prompt = RactoPrompt(
    role="You are a Python teacher.",
    aim="Explain the concept clearly.",
    constraints=["Use simple language.", "Give a short code example."],
    tone="Friendly",
    output_format="text",
)
kit = gpt.Chat(model="gpt-4o", default_prompt=prompt)

for chunk in kit.stream(gpt.ChatConfig(user_message="Explain Python generators")):
    print(chunk.delta.text, end="", flush=True)   # incremental text
    if chunk.is_final:
        print()
        print(f"Finish reason : {chunk.finish_reason}")
        print(f"Tokens used   : {chunk.usage}")
        print(f"Full response : {chunk.accumulated_text[:80]}...")
```

**Example output:**

```text
A generator in Python is a special function that yields values one at a time,
allowing you to iterate over a sequence without loading everything into memory.

def count_up(n):
    for i in range(n):
        yield i

for num in count_up(5):
    print(num)  # 0, 1, 2, 3, 4

Finish reason : FinishReason.STOP
Tokens used   : {"prompt_tokens": 55, "completion_tokens": 120, "total_tokens": 175}
Full response : A generator in Python is a special function that yields values one at a time...
```

### `StreamChunk` Field Reference

| Field | Type | Description |
| --- | --- | --- |
| `delta` | `StreamDelta` | Incremental content in this chunk |
| `accumulated_text` | `str` | Full text accumulated from all chunks so far |
| `is_final` | `bool` | `True` only on the very last chunk |
| `finish_reason` | `FinishReason \| None` | Set only on the final chunk |
| `tool_calls` | `list[ToolCallResult]` | Populated on the final chunk only (if tool calls occurred) |
| `usage` | `dict[str, int]` | Token usage — populated on the final chunk only |
| `raw` | `Any` | Raw provider streaming event |

### `StreamDelta` Field Reference

| Field | Type | Description |
| --- | --- | --- |
| `text` | `str` | Incremental text added in this chunk (empty string when no text) |
| `tool_call_id` | `str \| None` | Call ID of the tool call being streamed |
| `tool_call_name` | `str \| None` | Name of the tool being called |
| `tool_call_args_fragment` | `str \| None` | Partial JSON argument fragment |

---

## Async Support

Every method has a matching async variant.

```python
import asyncio
from ractogateway import openai_developer_kit as gpt, RactoPrompt

prompt = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Answer the user's question.",
    constraints=["Be concise."],
    tone="Friendly",
    output_format="text",
)
kit = gpt.Chat(model="gpt-4o", default_prompt=prompt)

async def main():
    # Async chat — returns LLMResponse
    response = await kit.achat(gpt.ChatConfig(user_message="What is SOLID?"))
    print(response.content)
    # "SOLID is a set of five object-oriented design principles: Single Responsibility,
    #  Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion."

    # Async streaming — yields StreamChunk
    async for chunk in kit.astream(gpt.ChatConfig(user_message="Explain SOLID briefly")):
        print(chunk.delta.text, end="", flush=True)
        if chunk.is_final:
            print(f"\nDone. Tokens: {chunk.usage}")

asyncio.run(main())
```

---

## Embeddings

### `EmbeddingConfig` — Input

```python
config = gpt.EmbeddingConfig(
    texts=["Hello world", "Goodbye world"],   # required — list of strings (min 1)
    model="text-embedding-3-large",            # optional (overrides kit default)
    dimensions=512,                            # optional — for models that support truncation
)
```

**`EmbeddingConfig` field reference:**

| Field | Type | Required | Default | Description |
| --- | --- | :---: | --- | --- |
| `texts` | `list[str]` | Yes | — | List of strings to embed (minimum 1) |
| `model` | `str \| None` | No | `None` | Override kit default embedding model |
| `dimensions` | `int \| None` | No | `None` | Output dimensionality (for supported models) |
| `extra` | `dict[str, Any]` | No | `{}` | Provider-specific pass-through kwargs |

### `EmbeddingResponse` — Output

```python
from ractogateway import openai_developer_kit as gpt

kit = gpt.Chat(model="gpt-4o", embedding_model="text-embedding-3-small")

response = kit.embed(gpt.EmbeddingConfig(texts=["cat", "dog", "automobile"]))

print(response.model)
# "text-embedding-3-small"

print(response.usage)
# {"prompt_tokens": 3, "total_tokens": 3}

print(len(response.vectors))
# 3

for v in response.vectors:
    print(f"[{v.index}] '{v.text}' → vector dim={len(v.embedding)}, first5={v.embedding[:5]}")
# [0] 'cat'        → vector dim=1536, first5=[0.023, -0.015, 0.041, ...]
# [1] 'dog'        → vector dim=1536, first5=[0.019, -0.012, 0.038, ...]
# [2] 'automobile' → vector dim=1536, first5=[-0.003, 0.027, -0.011, ...]
```

**`EmbeddingResponse` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `vectors` | `list[EmbeddingVector]` | One embedding per input text, in order |
| `model` | `str` | The model used for embedding |
| `usage` | `dict[str, int]` | `prompt_tokens`, `total_tokens` |
| `raw` | `Any` | Unmodified provider response |

**`EmbeddingVector` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `index` | `int` | 0-based position in the input `texts` list |
| `text` | `str` | The original input text |
| `embedding` | `list[float]` | The dense float vector |

---

## Tool Calling

Define tools as plain Python functions — never write nested JSON dicts by hand. RactoGateway translates them into the correct format for each provider.

### Register Tools with `@registry.register`

```python
from ractogateway import ToolRegistry

registry = ToolRegistry()

@registry.register
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    :param city: The city name
    :param unit: Temperature unit — celsius or fahrenheit
    """
    # Your real implementation here
    return f"Weather in {city}: 22°{unit[0].upper()}, partly cloudy"

@registry.register
def search_web(query: str, max_results: int = 3) -> list[str]:
    """Search the web for information.

    :param query: The search query
    :param max_results: Maximum number of results to return
    """
    return [f"Result {i}: ..." for i in range(1, max_results + 1)]
```

### Register Tools with the Standalone `@tool` Decorator

```python
from ractogateway import tool, ToolRegistry

@tool
def calculate_mortgage(
    principal: float,
    annual_rate: float,
    years: int,
) -> float:
    """Calculate monthly mortgage payment.

    :param principal: Loan amount in dollars
    :param annual_rate: Annual interest rate as a decimal (e.g., 0.05 for 5%)
    :param years: Loan term in years
    """
    monthly_rate = annual_rate / 12
    n = years * 12
    return principal * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)

# Then add the decorated function to a registry
registry = ToolRegistry()
registry.register(calculate_mortgage)
```

### Register Pydantic Models as Tools

```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    """Search the knowledge base for relevant documents."""
    query: str = Field(description="The search query string")
    max_results: int = Field(default=5, description="Maximum results to return")
    category: str = Field(default="all", description="Filter by category")

registry.register(SearchQuery)
```

### Use Tools with Any Kit

```python
config = gpt.ChatConfig(
    user_message="What's the weather in Tokyo and in Paris?",
    tools=registry,
)
response = kit.chat(config)

print(response.finish_reason)
# FinishReason.TOOL_CALL

for tc in response.tool_calls:
    print(f"Tool   : {tc.name}")
    print(f"Args   : {tc.arguments}")
    print(f"Call ID: {tc.id}")
    print()

# Tool   : get_weather
# Args   : {"city": "Tokyo", "unit": "celsius"}
# Call ID: call_abc123
#
# Tool   : get_weather
# Args   : {"city": "Paris", "unit": "celsius"}
# Call ID: call_def456

# Execute the tool and get the result
fn = registry.get_callable("get_weather")
result = fn(**response.tool_calls[0].arguments)
print(result)
# "Weather in Tokyo: 22°C, partly cloudy"
```

### `ToolRegistry` Method Reference

| Method / Property | Signature | Returns | Description |
| --- | --- | --- | --- |
| `register` | `(fn_or_model, name=None, description=None)` | `None` | Register a callable or Pydantic model as a tool |
| `schemas` | (property) | `list[ToolSchema]` | All registered tool schemas |
| `get_schema` | `(name: str)` | `ToolSchema \| None` | Look up a tool schema by name |
| `get_callable` | `(name: str)` | `Callable \| None` | Retrieve the original registered function |
| `__len__` | `len(registry)` | `int` | Total number of registered tools |
| `__contains__` | `name in registry` | `bool` | Check whether a tool name is registered |

### `ToolCallResult` Field Reference

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` | Provider-assigned call ID |
| `name` | `str` | Function name |
| `arguments` | `dict[str, Any]` | Parsed argument dict (ready to `**unpack`) |

### `ToolSchema` — Internal Schema Representation

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Tool name |
| `description` | `str` | Tool description |
| `parameters` | `list[ParamSchema]` | List of parameter descriptors |

**`ToolSchema` methods:**

| Method | Returns | Description |
| --- | --- | --- |
| `to_json_schema()` | `dict[str, Any]` | Produce OpenAI-compatible JSON Schema for the parameters |

---

## Validated Response Models

Force the LLM output into a specific Pydantic shape. If the model doesn't produce valid JSON matching your model, you get a clear validation error — not silent garbage.

```python
from pydantic import BaseModel
from ractogateway import openai_developer_kit as gpt, RactoPrompt

class SentimentResult(BaseModel):
    sentiment: str    # "positive", "negative", "neutral"
    confidence: float # 0.0 to 1.0
    reasoning: str    # short explanation

prompt = RactoPrompt(
    role="You are a sentiment analysis model.",
    aim="Classify the sentiment of the given text.",
    constraints=["Only classify as positive, negative, or neutral.", "Confidence must be between 0.0 and 1.0."],
    tone="Precise",
    output_format=SentimentResult,
)

kit = gpt.Chat(model="gpt-4o", default_prompt=prompt)

config = gpt.ChatConfig(
    user_message="Analyze sentiment: 'This product is absolutely amazing!'",
    response_model=SentimentResult,
)
response = kit.chat(config)

print(response.content)
# '{"sentiment": "positive", "confidence": 0.97, "reasoning": "Strong positive adjective 'amazing' with intensifier 'absolutely'."}'

print(response.parsed)
# {"sentiment": "positive", "confidence": 0.97, "reasoning": "Strong positive..."}

# Access as validated Pydantic object
result = SentimentResult(**response.parsed)
print(result.sentiment)    # "positive"
print(result.confidence)   # 0.97
print(result.reasoning)    # "Strong positive adjective 'amazing' with intensifier 'absolutely'."
```

---

## Multi-turn Conversations

Pass `history` to maintain context across turns.

```python
from ractogateway import openai_developer_kit as gpt, RactoPrompt

prompt = RactoPrompt(
    role="You are a helpful coding assistant.",
    aim="Help the user write and debug Python code.",
    constraints=["Always provide runnable code examples.", "Explain errors clearly."],
    tone="Friendly and educational",
    output_format="text",
)
kit = gpt.Chat(model="gpt-4o", default_prompt=prompt)

# Turn 1
r1 = kit.chat(gpt.ChatConfig(user_message="Write a function to reverse a string in Python."))
print(r1.content)
# "def reverse_string(s: str) -> str:\n    return s[::-1]"

# Turn 2 — pass history so the model remembers turn 1
r2 = kit.chat(gpt.ChatConfig(
    user_message="Now make it handle None input gracefully.",
    history=[
        gpt.Message(role=gpt.MessageRole.USER, content="Write a function to reverse a string in Python."),
        gpt.Message(role=gpt.MessageRole.ASSISTANT, content=r1.content),
    ],
))
print(r2.content)
# "def reverse_string(s: str | None) -> str | None:\n    if s is None:\n        return None\n    return s[::-1]"
```

---

## Multimodal Attachments

`RactoFile` lets you attach images, PDFs, plain-text files, and any binary file to a prompt. Use `prompt.to_messages()` to build provider-ready message lists that include the attachments in the correct format for each provider.

### Creating a `RactoFile`

```python
from ractogateway.prompts.engine import RactoFile

# From a file path — MIME type is auto-detected
img  = RactoFile.from_path("/path/to/photo.jpg")      # image/jpeg
doc  = RactoFile.from_path("/path/to/report.pdf")     # application/pdf
txt  = RactoFile.from_path("/path/to/notes.txt")      # text/plain

# From raw bytes — supply MIME type explicitly
with open("chart.png", "rb") as fh:
    chart = RactoFile.from_bytes(fh.read(), "image/png", name="chart.png")

# From a URL response
import requests
resp = requests.get("https://example.com/diagram.png")
diagram = RactoFile.from_bytes(resp.content, "image/png", name="diagram.png")
```

**`RactoFile` constructor methods:**

| Method | Signature | Returns | Description |
| --- | --- | --- | --- |
| `from_path` | `(path: str \| Path) -> RactoFile` | `RactoFile` | Load from file path; MIME auto-detected |
| `from_bytes` | `(data: bytes, mime_type: str, name: str) -> RactoFile` | `RactoFile` | Create from raw bytes |

**`RactoFile` property reference:**

| Member | Type | Description |
| --- | --- | --- |
| `data` | `bytes` | Raw file content |
| `mime_type` | `str` | MIME type, e.g. `"image/png"` |
| `name` | `str` | Filename hint |
| `base64_data` | `str` | Base-64 encoded file content |
| `is_image` | `bool` | `True` for JPEG, PNG, GIF, WebP |
| `is_pdf` | `bool` | `True` for `application/pdf` |
| `is_text` | `bool` | `True` for any `text/*` MIME |

### Building Multimodal Message Lists

Use `prompt.to_messages()` with the `attachments` parameter to build a multimodal message list, then pass it directly to the provider or low-level adapter:

```python
from ractogateway import RactoPrompt, Gateway
from ractogateway.adapters.openai_kit import OpenAILLMKit
from ractogateway.prompts.engine import RactoFile

prompt = RactoPrompt(
    role="You are a data analyst specialising in chart interpretation.",
    aim="Describe what the attached chart shows and extract the key insights.",
    constraints=[
        "Only describe what is visible in the image.",
        "Never invent data points not shown in the chart.",
    ],
    tone="Clear and concise",
    output_format="text",
)

# Build multimodal messages using to_messages()
attachment = RactoFile.from_path("sales_q4.png")
messages = prompt.to_messages(
    "What does this chart show?",
    attachments=[attachment],
    provider="openai",
)

# messages is now a list ready to send directly to the OpenAI API
# [
#   {"role": "system", "content": "<compiled RACTO prompt>"},
#   {"role": "user", "content": [
#       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
#       {"type": "text", "text": "What does this chart show?"}
#   ]}
# ]
```

### Provider Content-Block Translation

Each provider receives a different content-block format — `to_messages()` handles it transparently.

**OpenAI (`provider="openai"`)** — images become `image_url` blocks with inline data URIs:

```python
[
    {"role": "system", "content": "<compiled RACTO system prompt>"},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB..."}
            },
            {"type": "text", "text": "Describe the image."}
        ]
    }
]
```

**Anthropic (`provider="anthropic"`)** — images become `image` blocks, PDFs become `document` blocks:

```python
[
    {"role": "system", "content": "<compiled RACTO system prompt>"},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQSkZJRgAB..."}
            },
            {"type": "text", "text": "Describe the image."}
        ]
    }
]
```

**Google Gemini (`provider="google"`)** — files become `inline_data` parts:

```python
[
    {"role": "system", "content": "<compiled RACTO system prompt>"},
    {
        "role": "user",
        "content": [
            {"inline_data": {"mime_type": "image/jpeg", "data": "/9j/4AAQSkZJRgAB..."}},
            {"text": "Describe the image."}
        ]
    }
]
```

### Supported File Types

| File type | MIME type | OpenAI | Anthropic | Google |
| --- | --- | :---: | :---: | :---: |
| JPEG | `image/jpeg` | `image_url` | `image` block | `inline_data` |
| PNG | `image/png` | `image_url` | `image` block | `inline_data` |
| GIF | `image/gif` | `image_url` | `image` block | `inline_data` |
| WebP | `image/webp` | `image_url` | `image` block | `inline_data` |
| PDF | `application/pdf` | `image_url` (data URI) | `document` block | `inline_data` |
| Plain text | `text/plain` | `text` block | `text` block | `text` part |
| Any other | `*/*` | `image_url` (data URI) | labelled `text` block | `inline_data` |

---

## Low-Level Gateway

`Gateway` is a thin wrapper around any `BaseLLMAdapter`. Use it when you need direct access to prompt + adapter without the `ChatConfig` convenience layer — for example, when you want fine-grained control over individual calls.

### Creating and Using a Gateway

```python
from ractogateway import RactoPrompt, Gateway, ToolRegistry
from ractogateway.adapters.openai_kit import OpenAILLMKit

adapter = OpenAILLMKit(model="gpt-4o", api_key="sk-...")
prompt = RactoPrompt(
    role="You are a code reviewer.",
    aim="Identify bugs in the given code.",
    constraints=["Report only real bugs.", "If no bugs, say so."],
    tone="Concise",
    output_format="json",
)

gw = Gateway(adapter=adapter, default_prompt=prompt)

# Sync execution
response = gw.run(user_message="Review: def div(a, b): return a / b")
print(response.parsed)
# {"bugs": ["ZeroDivisionError if b is 0"], "severity": "high"}

# Async execution
import asyncio
async def main():
    response = await gw.arun(user_message="Review: x = 1; del x; print(x)")
    print(response.parsed)

asyncio.run(main())
```

**`Gateway` constructor parameters:**

| Parameter | Type | Required | Default | Description |
| --- | --- | :---: | --- | --- |
| `adapter` | `BaseLLMAdapter` | Yes | — | A concrete adapter (`OpenAILLMKit`, `GoogleLLMKit`, `AnthropicLLMKit`) |
| `tools` | `ToolRegistry \| None` | No | `None` | Default tool registry for all calls |
| `default_prompt` | `RactoPrompt \| None` | No | `None` | Fallback prompt when `run()` is called without one |

**`Gateway.run()` and `Gateway.arun()` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | `RactoPrompt \| None` | `None` | Override `default_prompt` for this call |
| `user_message` | `str` | `""` | The end-user's query |
| `tools` | `ToolRegistry \| None` | `None` | Override gateway-level tool registry |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_tokens` | `int` | `4096` | Maximum response tokens |
| `response_model` | `type[BaseModel] \| None` | `None` | Validate JSON output against this Pydantic model |
| `**kwargs` | `Any` | — | Passed through to the adapter |

**Returns:** `LLMResponse`

---

## Switching Providers

Same `ChatConfig`, different kit. Zero code changes to your prompt or config.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway import google_developer_kit as gemini
from ractogateway import anthropic_developer_kit as claude
from ractogateway import RactoPrompt

prompt = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Answer the user's question accurately.",
    constraints=["Be concise.", "Cite sources when possible."],
    tone="Friendly and professional",
    output_format="text",
)

config = gpt.ChatConfig(user_message="What is quantum computing?")

# OpenAI — use "gpt" alias
kit = gpt.Chat(model="gpt-4o", default_prompt=prompt)
print(kit.chat(config).content)
# "Quantum computing uses quantum bits (qubits) that can exist in superposition,
#  enabling calculations that classical computers cannot do efficiently..."

# Google Gemini — swap to "gemini" alias, everything else stays the same!
kit = gemini.Chat(model="gemini-2.0-flash", default_prompt=prompt)
print(kit.chat(config).content)
# "Quantum computing harnesses the principles of quantum mechanics..."

# Anthropic Claude — swap to "claude" alias, that's it!
kit = claude.Chat(model="claude-sonnet-4-6", default_prompt=prompt)
print(kit.chat(config).content)
# "Quantum computing is a type of computation that leverages quantum phenomena..."
```

---

## Fine-Tuning

RactoGateway includes a production-grade fine-tuning pipeline that works with OpenAI, Google Gemini, and Anthropic using a single, unified dataset API.

```python
from ractogateway import (
    RactoDataset,
    RactoTrainingExample,
    RactoTrainingMessage,
    OpenAIFineTuner,
    GeminiFineTuner,
    AnthropicFineTuner,
)
```

### Core Classes

| Class | Role |
| --- | --- |
| `RactoTrainingMessage` | One conversation turn — role + text + optional `RactoFile` attachments |
| `RactoTrainingExample` | One full training record (a conversation) — list of `RactoTrainingMessage` |
| `RactoDataset` | Collection of examples with validation, split, shuffle, and JSONL export |
| `OpenAIFineTuner` | Upload → create job → poll on OpenAI |
| `GeminiFineTuner` | Create tuning job → poll on Google AI |
| `AnthropicFineTuner` | Upload → create job → poll on Anthropic |

### `RactoTrainingMessage` Field Reference

| Field | Type | Required | Description |
| --- | --- | :---: | --- |
| `role` | `str` | Yes | `"system"`, `"user"`, or `"assistant"` |
| `content` | `str` | Yes | Text content of the message |
| `attachments` | `list[RactoFile]` | No | Optional multimodal file attachments |

**`RactoTrainingMessage` serialization methods:**

| Method | Returns | Description |
| --- | --- | --- |
| `to_openai()` | `dict` | Serialize to OpenAI message format |
| `to_anthropic()` | `dict` | Serialize to Anthropic message format |
| `to_gemini_parts()` | `list` | Serialize to Gemini content parts |

### `RactoTrainingExample` Factory Methods

| Factory Method | Signature | Description |
| --- | --- | --- |
| `from_pair` | `(user, assistant, system="", user_attachments=None)` | Single-turn from strings |
| `from_conversation` | `([(role, content), ...])` | Multi-turn from list of tuples |

**`RactoTrainingExample` serialization methods:**

| Method | Returns | Description |
| --- | --- | --- |
| `to_openai_dict()` | `dict` | OpenAI fine-tuning format |
| `to_anthropic_dict()` | `dict` | Anthropic fine-tuning format |
| `to_gemini_dict()` | `dict` | Gemini fine-tuning format |

### Step 1 — Build a Dataset

#### Quickest path — text pairs

```python
from ractogateway import RactoDataset

ds = RactoDataset.from_pairs(
    [
        ("What is a Python list?",  "An ordered, mutable sequence of items."),
        ("What is a Python dict?",  "An unordered key-value mapping."),
        ("What is a Python tuple?", "An ordered, immutable sequence."),
    ],
    system="You are a concise Python tutor. Answer in one sentence.",
)

print(ds.summary())
# {"examples": 3, "total_messages": 9, "avg_turns_per_example": 3.0, "multimodal_examples": 0}
```

#### Multi-turn conversation

```python
from ractogateway import RactoTrainingExample, RactoDataset

example = RactoTrainingExample.from_conversation([
    ("system",    "You are a helpful travel assistant."),
    ("user",      "I want to visit Japan. What season is best?"),
    ("assistant", "Spring (March–May) for cherry blossoms, or Autumn (Sept–Nov) for foliage."),
    ("user",      "Which cities should I visit?"),
    ("assistant", "Tokyo, Kyoto, Osaka, and Hiroshima are the most popular."),
])
ds = RactoDataset([example])
```

#### Multimodal example — image + text

```python
from ractogateway import RactoTrainingExample, RactoDataset
from ractogateway.prompts.engine import RactoFile

example = RactoTrainingExample.from_pair(
    user="Describe the trend shown in this chart.",
    assistant="Revenue grew by 23% quarter-over-quarter, peaking in December.",
    system="You are a data analyst. Be concise and factual.",
    user_attachments=[RactoFile.from_path("sales_chart.png")],
)
ds = RactoDataset([example])

print(ds.summary())
# {"examples": 1, "total_messages": 3, "avg_turns_per_example": 3.0, "multimodal_examples": 1}
```

#### Add examples incrementally

```python
ds = RactoDataset()
ds.add(RactoTrainingExample.from_pair("Q1", "A1", system="You are helpful."))
ds.add(RactoTrainingExample.from_pair("Q2", "A2", system="You are helpful."))
ds.extend([
    RactoTrainingExample.from_pair(u, a)
    for u, a in [("Q3", "A3"), ("Q4", "A4")]
])
```

### Step 2 — Validate and Split

```python
errors = ds.validate(provider="openai")   # or "anthropic" / "gemini"
if errors:
    for e in errors:
        print(e)
else:
    print("Dataset is valid.")

# Reproducible 80/20 train-validation split
train_ds, val_ds = ds.split(train_ratio=0.8, seed=42)
print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")
# Train: 80  |  Val: 20
```

### Step 3 — Export to JSONL (optional inspection)

```python
train_ds.export_jsonl("train.jsonl",     provider="openai",    overwrite=True)
val_ds.export_jsonl("val.jsonl",         provider="openai",    overwrite=True)
train_ds.export_jsonl("train_ant.jsonl", provider="anthropic", overwrite=True)
train_ds.export_jsonl("train_gem.jsonl", provider="gemini",    overwrite=True)
```

**OpenAI JSONL format** (`train.jsonl`):

```json
{"messages": [{"role": "system", "content": "You are a Python tutor."}, {"role": "user", "content": "What is a list?"}, {"role": "assistant", "content": "An ordered, mutable sequence."}]}
{"messages": [{"role": "system", "content": "You are a Python tutor."}, {"role": "user", "content": "What is a dict?"}, {"role": "assistant", "content": "A key-value mapping."}]}
```

**Anthropic JSONL format** (`train_ant.jsonl`):

```json
{"system": "You are a Python tutor.", "messages": [{"role": "user", "content": "What is a list?"}, {"role": "assistant", "content": "An ordered, mutable sequence."}]}
```

**Gemini JSONL format** (`train_gem.jsonl`):

```json
{"text_input": "What is a list?", "output": "An ordered, mutable sequence."}
```

**OpenAI multimodal format** (image in user turn):

```json
{
  "messages": [
    {"role": "system", "content": "You are a data analyst."},
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
        {"type": "text", "text": "Describe the trend."}
      ]
    },
    {"role": "assistant", "content": "Revenue grew 23% quarter-over-quarter."}
  ]
}
```

### Step 4 — Fine-Tune

#### OpenAI — one call

```python
from ractogateway import OpenAIFineTuner

tuner = OpenAIFineTuner(api_key="sk-...")   # or set OPENAI_API_KEY

fine_tuned_model = tuner.run_pipeline(
    train_ds,
    model="gpt-4o-mini-2024-07-18",
    validation_dataset=val_ds,
    n_epochs=3,
    suffix="python-tutor",
    verbose=True,
)
# [OpenAIFineTuner] Uploading 80 training examples…
# [OpenAIFineTuner] Training file: file-abc123
# [OpenAIFineTuner] Job created: ftjob-xyz789
# [OpenAIFineTuner] Job ftjob-xyz789 → running
# [OpenAIFineTuner] Done!  Fine-tuned model: ft:gpt-4o-mini-2024-07-18:org::python-tutor-abc

# Use immediately
from ractogateway import openai_developer_kit as gpt
kit = gpt.Chat(model=fine_tuned_model)
response = kit.chat(gpt.ChatConfig(user_message="What is a generator?"))
print(response.content)
# "A generator is a function that uses yield to produce values lazily, one at a time."
```

#### OpenAI — step by step

```python
tuner = OpenAIFineTuner()

train_file_id = tuner.upload_dataset(train_ds)
val_file_id   = tuner.upload_dataset(val_ds)

job_id = tuner.create_job(
    train_file_id,
    model="gpt-4o-mini-2024-07-18",
    validation_file=val_file_id,
    n_epochs=3,
    suffix="python-tutor",
)

print(tuner.get_status(job_id))
# {"id": "ftjob-…", "status": "running", "model": "gpt-4o-mini-2024-07-18", ...}

for event in tuner.list_events(job_id, limit=10):
    print(event["message"])

fine_tuned_model = tuner.wait_for_completion(job_id, poll_interval=30)
```

**`OpenAIFineTuner` method reference:**

| Method | Signature | Returns | Description |
| --- | --- | --- | --- |
| `run_pipeline` | `(train_ds, model, validation_dataset=None, n_epochs=3, suffix="", verbose=False)` | `str` | Full pipeline — upload, create job, wait, return model name |
| `upload_dataset` | `(ds: RactoDataset)` | `str` | Upload dataset, return file ID |
| `create_job` | `(train_file_id, model, validation_file=None, n_epochs=3, suffix="")` | `str` | Create fine-tune job, return job ID |
| `get_status` | `(job_id: str)` | `dict` | Get current job status |
| `list_events` | `(job_id: str, limit=10)` | `list[dict]` | Get recent job events |
| `wait_for_completion` | `(job_id: str, poll_interval=30)` | `str` | Poll until done, return fine-tuned model name |

#### Google Gemini — one call

```python
from ractogateway import GeminiFineTuner

tuner = GeminiFineTuner(api_key="AIza...")

tuned_model = tuner.run_pipeline(
    train_ds,
    base_model="models/gemini-1.5-flash-001-tuning",
    display_name="python-tutor",
    epoch_count=5,
    batch_size=4,
    verbose=True,
)
# [GeminiFineTuner] Starting tuning with 80 examples…
# [GeminiFineTuner] State: CREATING (12%)
# [GeminiFineTuner] Done!  Tuned model: tunedModels/python-tutor-abc123

from ractogateway import google_developer_kit as gemini
kit = gemini.Chat(model=tuned_model)
```

#### Anthropic Claude — one call

```python
from ractogateway import AnthropicFineTuner

tuner = AnthropicFineTuner(api_key="sk-ant-...")

fine_tuned_model = tuner.run_pipeline(
    train_ds,
    model="claude-3-haiku-20240307",
    validation_dataset=val_ds,
    suffix="python-tutor",
    hyperparameters={"n_epochs": 3},
    verbose=True,
)
# [AnthropicFineTuner] Uploading 80 training examples…
# [AnthropicFineTuner] Training file: file-…
# [AnthropicFineTuner] Job created: ftjob-…
# [AnthropicFineTuner] Done!  Fine-tuned model: claude-3-haiku-20240307:ft:…
```

### `RactoDataset` API Reference

| Member | Signature | Returns | Description |
| --- | --- | --- | --- |
| `RactoDataset.from_pairs` | `(pairs, system="")` | `RactoDataset` | Build from `[(user, assistant)]` text tuples |
| `RactoDataset.from_jsonl` | `(path, provider="openai")` | `RactoDataset` | Load a previously exported JSONL file |
| `.add` | `(example: RactoTrainingExample)` | `None` | Append one example |
| `.extend` | `(examples: list)` | `None` | Append a list of examples |
| `.validate` | `(provider: str)` | `list[str]` | Returns list of errors (empty = valid) |
| `.split` | `(train_ratio=0.8, seed=42)` | `(RactoDataset, RactoDataset)` | Reproducible train/val split |
| `.shuffle` | `(seed: int)` | `RactoDataset` | Returns a new shuffled dataset |
| `.export_jsonl` | `(path, provider, overwrite=True)` | `None` | Write to `.jsonl` file on disk |
| `.to_jsonl_string` | `(provider: str)` | `str` | Return JSONL as a string (no I/O) |
| `.summary` | `()` | `dict` | Stats: `examples`, `total_messages`, `multimodal_examples`, … |

### Provider Fine-Tuning Support Matrix

| Feature | OpenAI | Gemini | Anthropic |
| --- | :---: | :---: | :---: |
| Text-only fine-tuning | Yes | Yes | Yes |
| Multimodal (image) fine-tuning | Yes (`gpt-4o-2024-08-06`) | Vertex AI only | Yes |
| Multi-turn conversations | Yes | Vertex AI only | Yes |
| Validation dataset | Yes | No | Yes |
| Hyperparameter control | epochs, batch, LR | epochs, batch, LR | epochs |
| `run_pipeline()` one-liner | Yes | Yes | Yes |

---

## RAG

RactoGateway ships a full Retrieval-Augmented Generation (RAG) pipeline. In plain English: you feed it documents, it breaks them into chunks, converts them to number vectors, stores them, and later retrieves the most relevant chunks to answer a question — all in one class.

```text
Document → Read → Chunk → Process → Embed → Store
                                              ↓
                              Query → Embed → Retrieve → Generate → Answer
```

### RAG Installation

```bash
pip install ractogateway[rag-all]    # everything
# or pick what you need:
pip install ractogateway[rag]        # base readers + NLP
pip install ractogateway[rag-pdf]    # PDF
pip install ractogateway[rag-chroma] # ChromaDB
```

### Quickstart — 4 Lines

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.rag.pipeline import RactoRAG
from ractogateway.rag.embedders import OpenAIEmbedder
from ractogateway.rag.stores import InMemoryVectorStore

kit = gpt.Chat(model="gpt-4o")
rag = RactoRAG(
    vector_store=InMemoryVectorStore(),
    embedder=OpenAIEmbedder(),
    llm_kit=kit,
)
rag.ingest("report.pdf")
response = rag.query("What were the key findings?")
print(response.answer.content)
# "The key findings were: (1) revenue increased 22% YoY, (2) customer churn
#  dropped by 4 percentage points, (3) the APAC region became the fastest-growing market."
```

### `RactoRAG` Constructor Parameters

| Parameter | Type | Required | Default | Description |
| --- | --- | :---: | --- | --- |
| `vector_store` | `BaseVectorStore` | Yes | — | Where chunks are indexed and searched |
| `embedder` | `BaseEmbedder` | Yes | — | Converts text to float vectors |
| `chunker` | `BaseChunker \| None` | No | `RecursiveChunker(512, 50)` | How documents are split |
| `processors` | `list[BaseProcessor] \| None` | No | `[TextCleaner()]` | Text cleaning pipeline |
| `llm_kit` | `Any \| None` | No* | `None` | Required for `.query()` / `.aquery()` |
| `context_template` | `str \| None` | No | Built-in | Template for injecting context into the LLM |
| `reader_registry` | `FileReaderRegistry \| None` | No | Built-in | Dispatches files to the correct reader |
| `default_prompt` | `RactoPrompt \| None` | No | Built-in RAG prompt | System prompt used during generation |

> \* `llm_kit` is optional at construction time but required when calling `.query()` or `.aquery()`.

### Ingesting Documents

```python
# Single file (auto-detected reader based on extension)
chunks = rag.ingest("report.pdf")
chunks = rag.ingest("notes.txt")
chunks = rag.ingest("data.xlsx")
chunks = rag.ingest("page.html")

print(len(chunks))
# 47   ← number of chunks created from the document

print(chunks[0])
# Chunk(
#   chunk_id="3f8a2c1d-...",
#   doc_id="a1b2c3d4-...",
#   content="The annual report shows revenue growth of 22%...",
#   embedding=[0.023, -0.015, 0.041, ...],  # 1536-dim vector
#   metadata=ChunkMetadata(
#       source="/path/to/report.pdf",
#       page=1,
#       chunk_index=0,
#       total_chunks=47,
#       start_char=0,
#       end_char=512,
#       doc_id="a1b2c3d4-...",
#       extra={}
#   )
# )

# Entire directory (recursively, all supported file types)
chunks = rag.ingest_dir("./docs/", pattern="**/*.pdf")

# Raw text string — no file needed
chunks = rag.ingest_text(
    "The quick brown fox jumps over the lazy dog.",
    source="manual-input",
    category="test",    # extra metadata
)

# Async variants
chunks = await rag.aingest("big_report.pdf")
chunks = await rag.aingest_dir("./docs/")
chunks = await rag.aingest_text("some text", source="api")
```

**`ingest()` / `aingest()` parameters:**

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `str \| Path` | File path to ingest |
| `**metadata` | `Any` | Extra key-value pairs stored in `ChunkMetadata.extra` |

**Returns:** `list[Chunk]`

**`ingest_dir()` / `aingest_dir()` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `directory` | `str \| Path` | — | Directory to walk |
| `pattern` | `str` | `"**/*"` | Glob pattern to filter files |
| `**metadata` | `Any` | — | Extra metadata attached to all chunks |

**Returns:** `list[Chunk]`

**`ingest_text()` / `aingest_text()` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `str` | — | Raw text content to ingest |
| `source` | `str` | `"manual"` | Label for this text source |
| `**metadata` | `Any` | — | Extra metadata attached to all chunks |

**Returns:** `list[Chunk]`

### RAG Data Models

**`Document` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `doc_id` | `str` | Auto-generated UUID for this document |
| `content` | `str` | Full extracted text content |
| `source` | `str` | File path, URL, or caller-supplied label |
| `metadata` | `dict[str, Any]` | Arbitrary metadata dict |

**`Chunk` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `chunk_id` | `str` | Auto-generated UUID for this chunk |
| `doc_id` | `str` | UUID of the parent `Document` |
| `content` | `str` | Text content of this chunk |
| `embedding` | `list[float] \| None` | Dense float vector (`None` until embedded) |
| `metadata` | `ChunkMetadata` | Provenance info for this chunk |

**`ChunkMetadata` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `source` | `str` | File path or URL |
| `page` | `int \| None` | Page number for PDFs (1-based), else `None` |
| `chunk_index` | `int` | 0-based position within the parent document |
| `total_chunks` | `int` | Total chunks created from the parent document |
| `start_char` | `int` | Character offset where this chunk starts |
| `end_char` | `int` | Character offset where this chunk ends |
| `doc_id` | `str` | UUID of the parent document |
| `extra` | `dict[str, Any]` | Caller-supplied metadata (from `ingest(**metadata)`) |

### Retrieving Without Generating

```python
results = rag.retrieve("What is the revenue growth?", top_k=3)

for r in results:
    print(f"Rank {r.rank} | Score {r.score:.4f} | Source: {r.chunk.metadata.source}")
    print(f"  {r.chunk.content[:100]}...")
    print()

# Rank 1 | Score 0.9231 | Source: /path/to/report.pdf
#   The company achieved revenue growth of 22% year-over-year, driven by...
#
# Rank 2 | Score 0.8847 | Source: /path/to/report.pdf
#   In FY2024, total revenue reached $12.4 million, compared to $10.2 million...
#
# Rank 3 | Score 0.8102 | Source: /path/to/report.pdf
#   The APAC region contributed most significantly to revenue growth, with...
```

**`retrieve()` / `aretrieve()` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `str` | — | The search query text |
| `top_k` | `int` | `5` | Maximum number of results to return |
| `filters` | `dict \| None` | `None` | Metadata filters (store-specific) |

**Returns:** `list[RetrievalResult]`

**`RetrievalResult` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `chunk` | `Chunk` | The retrieved text chunk |
| `score` | `float` | Similarity score (higher = more relevant) |
| `rank` | `int` | 1-based rank (1 = most relevant) |

### Full RAG Query — Retrieve + Generate

```python
rag_response = rag.query(
    "What is the revenue growth and which region performed best?",
    top_k=5,           # retrieve 5 most relevant chunks
    temperature=0.0,   # factual answers — keep temperature low
    max_tokens=2048,
)

print(rag_response.answer.content)
# "Based on the provided context:
#  1. Revenue grew 22% year-over-year, reaching $12.4M in FY2024.
#  2. The APAC region was the top performer, contributing significantly to growth.
#  Source: report.pdf (page 3)"

print(f"Query  : {rag_response.query}")
# Query  : What is the revenue growth and which region performed best?

print(f"Sources: {len(rag_response.sources)}")
# Sources: 5

for r in rag_response.sources:
    print(f"  [{r.rank}] score={r.score:.3f} → {r.chunk.content[:60]}...")
# [1] score=0.923 → The company achieved revenue growth of 22% year-over-year...
# [2] score=0.885 → In FY2024, total revenue reached $12.4 million...
# [3] score=0.810 → The APAC region contributed most significantly...
# [4] score=0.776 → North America remained the largest single market...
# [5] score=0.741 → EMEA recorded moderate growth of 9% year-over-year...
```

**`query()` / `aquery()` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `question` | `str` | — | The user's question (required) |
| `top_k` | `int` | `5` | Chunks to retrieve and inject as context |
| `filters` | `dict \| None` | `None` | Metadata filters (store-specific) |
| `prompt` | `RactoPrompt \| None` | `None` | Override default RAG prompt |
| `temperature` | `float` | `0.0` | Sampling temperature for generation |
| `max_tokens` | `int` | `2048` | Maximum tokens in the generated answer |

**Returns:** `RAGResponse`

**`RAGResponse` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `answer` | `LLMResponse` | The generated answer (same as a normal `chat()` response) |
| `sources` | `list[RetrievalResult]` | Chunks used as context for generation |
| `query` | `str` | The original question |
| `context_used` | `str` | Verbatim context string injected into the LLM |

### Async RAG

```python
chunks = await rag.aingest("big_report.pdf")
results = await rag.aretrieve("key findings", top_k=3)
response = await rag.aquery("What were the key findings?")
print(response.answer.content)
```

### RAG — Chunking Strategies

```python
from ractogateway.rag.chunkers import (
    FixedChunker,      # Split at exactly N characters
    RecursiveChunker,  # Smart split on paragraphs → sentences → words (default)
    SentenceChunker,   # Split on sentence boundaries
    SemanticChunker,   # Split where meaning changes (requires embedder)
)

# Fixed — simple, predictable
rag = RactoRAG(
    vector_store=InMemoryVectorStore(),
    embedder=OpenAIEmbedder(),
    chunker=FixedChunker(chunk_size=256, overlap=32),
    llm_kit=kit,
)

# Recursive — good default, respects paragraph/sentence structure
rag = RactoRAG(
    vector_store=InMemoryVectorStore(),
    embedder=OpenAIEmbedder(),
    chunker=RecursiveChunker(chunk_size=512, overlap=50),
    llm_kit=kit,
)

# Sentence — split on natural sentence boundaries
rag = RactoRAG(
    vector_store=InMemoryVectorStore(),
    embedder=OpenAIEmbedder(),
    chunker=SentenceChunker(max_sentences=5),
    llm_kit=kit,
)

# Semantic — split where meaning changes (requires an embedder reference)
from ractogateway.rag.chunkers import SemanticChunker
embedder = OpenAIEmbedder()
rag = RactoRAG(
    vector_store=InMemoryVectorStore(),
    embedder=embedder,
    chunker=SemanticChunker(embedder=embedder, threshold=0.8),
    llm_kit=kit,
)
```

**Chunker parameter reference:**

| Chunker | Key Parameters | Description |
| --- | --- | --- |
| `FixedChunker` | `chunk_size=256`, `overlap=32` | Split at exactly `chunk_size` characters with `overlap` overlap |
| `RecursiveChunker` | `chunk_size=512`, `overlap=50` | Hierarchical: paragraphs → sentences → words |
| `SentenceChunker` | `max_sentences=5` | Split every `max_sentences` sentence boundaries |
| `SemanticChunker` | `embedder`, `threshold=0.8` | Split where cosine similarity drops below `threshold` |

**`BaseChunker` interface:**

| Method | Signature | Returns | Description |
| --- | --- | --- | --- |
| `chunk` | `(document: Document) -> list[Chunk]` | `list[Chunk]` | Split a document into chunks |

### RAG — Embedders

```python
from ractogateway.rag.embedders import OpenAIEmbedder, GoogleEmbedder, VoyageEmbedder

# OpenAI
embedder = OpenAIEmbedder(
    model="text-embedding-3-small",   # default
    api_key="sk-...",                 # or OPENAI_API_KEY
)

# Google
embedder = GoogleEmbedder(
    model="text-embedding-004",        # default
    api_key="AIza...",                 # or GEMINI_API_KEY
)

# Voyage AI (great for RAG)
embedder = VoyageEmbedder(
    model="voyage-3",
    api_key="pa-...",
)
```

**`BaseEmbedder` interface:**

| Method / Property | Signature | Returns | Description |
| --- | --- | --- | --- |
| `dimension` | (property) | `int` | Embedding dimension size (`-1` if unknown before first call) |
| `embed` | `(texts: list[str]) -> list[list[float]]` | `list[list[float]]` | Synchronous batch embedding |
| `aembed` | `(texts: list[str]) -> list[list[float]]` | `list[list[float]]` | Async batch embedding |

### RAG — Vector Stores

```python
from ractogateway.rag.stores import (
    InMemoryVectorStore,   # no setup, great for prototyping
    ChromaStore,           # pip install ractogateway[rag-chroma]
    FAISSStore,            # pip install ractogateway[rag-faiss]
    PineconeStore,         # pip install ractogateway[rag-pinecone]
    QdrantStore,           # pip install ractogateway[rag-qdrant]
    WeaviateStore,         # pip install ractogateway[rag-weaviate]
    MilvusStore,           # pip install ractogateway[rag-milvus]
    PGVectorStore,         # pip install ractogateway[rag-pgvector]
)

# In-memory (no setup)
store = InMemoryVectorStore()

# ChromaDB (local persistence)
store = ChromaStore(collection="my_docs", persist_directory="./chroma_db")

# FAISS (fast local search)
store = FAISSStore(index_path="./faiss.index", dimension=1536)

# Pinecone (cloud)
store = PineconeStore(index_name="my-index", api_key="...")

# Qdrant (self-hosted or cloud)
store = QdrantStore(collection="my_docs", url="http://localhost:6333")

# PostgreSQL pgvector
store = PGVectorStore(connection_string="postgresql://user:pass@localhost/db", table="embeddings")
```

**`BaseVectorStore` interface:**

| Method | Signature | Returns | Description |
| --- | --- | --- | --- |
| `add` | `(chunks: list[Chunk]) -> None` | `None` | Index chunks (must have embeddings set) |
| `search` | `(embedding: list[float], top_k=5, filters=None) -> list[RetrievalResult]` | `list[RetrievalResult]` | Find most similar chunks |
| `delete` | `(chunk_ids: list[str]) -> None` | `None` | Remove chunks by ID |
| `clear` | `() -> None` | `None` | Remove all indexed chunks |
| `count` | `() -> int` | `int` | Total indexed chunk count |

### RAG — Readers

Documents are loaded automatically based on file extension:

| Reader | Extensions | Install |
| --- | --- | --- |
| `TextReader` | `.txt`, `.md`, `.rst`, `.csv` | Built-in |
| `HtmlReader` | `.html`, `.htm` | Built-in |
| `PdfReader` | `.pdf` | `ractogateway[rag-pdf]` |
| `WordReader` | `.docx` | `ractogateway[rag-word]` |
| `SpreadsheetReader` | `.xlsx`, `.xls` | `ractogateway[rag-excel]` |
| `ImageReader` | `.jpg`, `.jpeg`, `.png`, `.gif` | `ractogateway[rag-image]` |

**`BaseReader` interface:**

| Method / Property | Signature | Returns | Description |
| --- | --- | --- | --- |
| `supported_extensions` | (property) | `frozenset[str]` | File extensions this reader handles |
| `read` | `(path: Path) -> Document` | `Document` | Load a file and return a `Document` |

### RAG — File Reader Registry

`FileReaderRegistry` auto-dispatches file reads to the correct reader based on extension.

```python
from ractogateway import FileReaderRegistry
from ractogateway.rag.readers import TextReader, PdfReader

# The registry used by RactoRAG is built-in (auto-registers all available readers)
# You can also create a custom one:
registry = FileReaderRegistry()
registry.register(TextReader())     # manually register a reader
registry.register(PdfReader())

# Read a file — dispatches automatically
doc = registry.read("report.pdf")  # → Document
print(doc.content)                  # extracted text
print(doc.source)                   # "report.pdf"
```

**`FileReaderRegistry` method reference:**

| Method | Signature | Returns | Description |
| --- | --- | --- | --- |
| `register` | `(reader: BaseReader) -> None` | `None` | Add a reader for its `supported_extensions` |
| `read` | `(path: str \| Path) -> Document` | `Document` | Auto-dispatch to the matching reader |
| `can_read` | `(path: str \| Path) -> bool` | `bool` | Check if any reader handles this extension |

### RAG — Processing Pipeline

Text processors clean and normalise chunks before embedding:

```python
from ractogateway.rag.processors import TextCleaner, Lemmatizer, ProcessingPipeline

rag = RactoRAG(
    vector_store=InMemoryVectorStore(),
    embedder=OpenAIEmbedder(),
    processors=[
        TextCleaner(),    # strip extra whitespace, fix encoding
        Lemmatizer(),     # reduce words to root form (pip install ractogateway[rag-nlp])
    ],
    llm_kit=kit,
)

# ProcessingPipeline chains multiple processors manually
pipeline = ProcessingPipeline([TextCleaner(), Lemmatizer()])
cleaned_text = pipeline.process("  Running   quickly through the fields...  ")
# "run quickly through the field"
```

**`BaseProcessor` interface:**

| Method | Signature | Returns | Description |
| --- | --- | --- | --- |
| `process` | `(text: str) -> str` | `str` | Transform text and return cleaned result |

**`ProcessingPipeline` — chains processors:**

| Method | Signature | Returns | Description |
| --- | --- | --- | --- |
| `__init__` | `(processors: list[BaseProcessor])` | — | Build the pipeline |
| `process` | `(text: str) -> str` | `str` | Run text through all processors in order |

### Full RAG Pipeline Example — Production Setup

```python
from ractogateway import openai_developer_kit as gpt, RactoPrompt
from ractogateway.rag.pipeline import RactoRAG
from ractogateway.rag.embedders import OpenAIEmbedder
from ractogateway.rag.stores import ChromaStore
from ractogateway.rag.chunkers import RecursiveChunker
from ractogateway.rag.processors import TextCleaner

# 1. Build the kit
kit = gpt.Chat(model="gpt-4o")

# 2. Custom RAG prompt
rag_prompt = RactoPrompt(
    role="You are a precise document Q&A assistant.",
    aim="Answer the user's question using only the provided context.",
    constraints=[
        "Never fabricate information not in the context.",
        "If the context doesn't contain the answer, say so clearly.",
        "Cite the source document and page number when available.",
    ],
    tone="Professional and concise",
    output_format="text",
)

# 3. Assemble the pipeline
rag = RactoRAG(
    vector_store=ChromaStore(collection="company_docs", persist_directory="./db"),
    embedder=OpenAIEmbedder(model="text-embedding-3-large"),
    chunker=RecursiveChunker(chunk_size=512, overlap=64),
    processors=[TextCleaner()],
    llm_kit=kit,
    default_prompt=rag_prompt,
)

# 4. Ingest your document library
total_chunks = rag.ingest_dir("./company_docs/", pattern="**/*.pdf")
print(f"Indexed {rag.count()} chunks from {len(total_chunks)} files")
# Indexed 1247 chunks from 23 files

# 5. Answer questions
response = rag.query("What is our refund policy for digital products?", top_k=5)

print(response.answer.content)
# "According to the company policy document (page 4):
#  Digital products are eligible for a full refund within 14 days of purchase,
#  provided the product has not been downloaded more than 3 times.
#  After 14 days, refunds are issued as store credit only."

print(f"\nContext came from {len(response.sources)} sources:")
for r in response.sources:
    src = r.chunk.metadata.source.split("/")[-1]
    pg  = f", page {r.chunk.metadata.page}" if r.chunk.metadata.page else ""
    print(f"  [{r.rank}] {src}{pg} (score={r.score:.3f})")
# [1] refund_policy.pdf, page 4 (score=0.941)
# [2] refund_policy.pdf, page 5 (score=0.882)
# [3] customer_handbook.pdf, page 12 (score=0.791)
# [4] faq.pdf, page 2 (score=0.743)
# [5] terms_of_service.pdf, page 7 (score=0.701)
```

### `RactoRAG` Method Reference

| Method | Signature | Returns | Description |
| --- | --- | --- | --- |
| `ingest` | `(path, **metadata)` | `list[Chunk]` | Read, chunk, embed, and store a file |
| `ingest_dir` | `(directory, pattern="**/*", **metadata)` | `list[Chunk]` | Recursively ingest all supported files |
| `ingest_text` | `(text, source="manual", **metadata)` | `list[Chunk]` | Ingest raw text directly |
| `aingest` | `(path, **metadata)` | `list[Chunk]` | Async variant of `ingest` |
| `aingest_dir` | `(directory, pattern, **metadata)` | `list[Chunk]` | Async variant of `ingest_dir` |
| `aingest_text` | `(text, source, **metadata)` | `list[Chunk]` | Async variant of `ingest_text` |
| `retrieve` | `(query, top_k=5, filters=None)` | `list[RetrievalResult]` | Embed query and return top-k chunks |
| `aretrieve` | `(query, top_k=5, filters=None)` | `list[RetrievalResult]` | Async variant of `retrieve` |
| `query` | `(question, top_k=5, filters=None, prompt=None, temperature=0.0, max_tokens=2048)` | `RAGResponse` | Retrieve + generate → full RAG answer |
| `aquery` | `(...)` | `RAGResponse` | Async variant of `query` |
| `count` | `()` | `int` | Total indexed chunks |
| `clear` | `()` | `None` | Remove all indexed chunks |
| `store` | (property) | `BaseVectorStore` | Access the underlying vector store |
| `embedder` | (property) | `BaseEmbedder` | Access the underlying embedder |

---

## Performance & Cost Optimization

Five production-grade features that reduce latency, token spend, and API cost — all optional, zero-cost when not used, and available on every developer kit.

| Feature | What it does | Cost saving |
| --- | --- | --- |
| **Exact-match cache** | Returns cached response for identical requests (SHA-256 key) | 100 % API cost for repeats |
| **Semantic cache** | Returns cached response for semantically similar queries | 100 % API cost for near-duplicates |
| **Cost-aware routing** | Picks cheapest model based on request complexity | 50–90 % on simple requests |
| **Token truncation** | Trims history before context-window overflow | Prevents 400 errors + wasted tokens |
| **Batch processing** | Queues thousands of tasks via provider Batch APIs | ~50 % off standard API pricing |

All four middleware features (`exact_cache`, `semantic_cache`, `router`, `truncator`) are optional constructor parameters on every kit. None of them are active unless you pass them in.

> **Multi-server deployments?** See the [Redis Infrastructure](#redis-infrastructure) section for distributed versions of the exact-match cache, rate limiter, and chat memory that work across an entire fleet.

---

### Exact-Match Cache

An in-memory LRU cache keyed on `SHA-256(user_message + system_prompt + model + temperature + max_tokens)`. Identical requests return instantly — no API call, no latency, no cost.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.cache import ExactMatchCache

kit = gpt.Chat(
    model="gpt-4o",
    default_prompt=prompt,
    exact_cache=ExactMatchCache(max_size=1024, ttl_seconds=3600),  # 1 h TTL, 1024 entries
)

config = gpt.ChatConfig(user_message="What is the capital of France?")

r1 = kit.chat(config)
print(r1.content)
# "The capital of France is Paris."

r2 = kit.chat(config)   # identical request → served from cache
print(r2.content)
# "The capital of France is Paris."    ← same answer, 0 ms, $0.00

# Inspect cache performance
stats = kit.exact_cache.stats
print(stats)
# CacheStats(hits=1, misses=1, size=1, hit_rate=50.0%)

print(stats.hits)        # 1
print(stats.misses)      # 1
print(stats.hit_rate)    # 0.5
print(stats.size)        # 1   (entries stored)
```

**`ExactMatchCache` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_size` | `int` | `1024` | Max entries before LRU eviction. `0` = unlimited |
| `ttl_seconds` | `float \| None` | `None` | Seconds before an entry expires. `None` = never |

**Cache methods:**

| Method | Description |
| --- | --- |
| `get(key)` | Returns `LLMResponse` if hit, `None` if miss |
| `put(key, response)` | Store a response |
| `invalidate(key)` | Remove one entry |
| `clear()` | Flush all entries |
| `stats` | Returns `CacheStats(hits, misses, size)` |

---

### Semantic Cache

Embeds the query and compares against stored embeddings using cosine similarity. If any stored query is ≥ `threshold` similar, the cached answer is returned — no API call needed. You wire in any embedding function you like (your RAG embedder, OpenAI embeddings, etc.).

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.cache import SemanticCache

# Wire in any embedding function: Callable[[str], list[float]]
def my_embedder(text: str) -> list[float]:
    resp = gpt.Chat(model="gpt-4o").embed(
        gpt.EmbeddingConfig(texts=[text])
    )
    return resp.vectors[0].vector

kit = gpt.Chat(
    model="gpt-4o",
    default_prompt=prompt,
    semantic_cache=SemanticCache(
        embedder=my_embedder,
        threshold=0.95,   # 95 % cosine similarity → cache hit
        max_size=512,
        ttl_seconds=1800,
    ),
)

r1 = kit.chat(gpt.ChatConfig(user_message="What is the capital of France?"))
print(r1.content)
# "The capital of France is Paris."

# Semantically equivalent — slightly different phrasing
r2 = kit.chat(gpt.ChatConfig(user_message="Which city is the capital of France?"))
print(r2.content)
# "The capital of France is Paris."   ← cache hit, cosine sim ≥ 0.95

stats = kit.semantic_cache.stats
print(stats)
# CacheStats(hits=1, misses=1, size=1, hit_rate=50.0%)
```

**`SemanticCache` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `embedder` | `Callable[[str], list[float]]` | required | Any function that returns a float vector for a text |
| `threshold` | `float` | `0.95` | Minimum cosine similarity (0.0–1.0) to declare a hit |
| `max_size` | `int` | `512` | Max entries before LRU eviction. `0` = unlimited |
| `ttl_seconds` | `float \| None` | `None` | Seconds before an entry expires. `None` = never |

> **Tip:** You can use the RAG embedders directly: `from ractogateway.rag.embedders import OpenAIEmbedder` — call `embedder.embed([text])` and return `result[0]`.

---

### Cost-Aware Routing

Set `model="auto"` and provide a `CostAwareRouter` with an ordered tier list. Each incoming message receives a complexity score (0–100) based on estimated token length and keyword analysis. The router picks the **first tier** whose `max_score` covers the score — so simple messages go to cheap models automatically.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.routing import CostAwareRouter, RoutingTier

# Define tiers sorted cheapest → most capable (ascending max_score)
router = CostAwareRouter(tiers=[
    RoutingTier(model="gpt-4o-mini", max_score=30),   # short / simple
    RoutingTier(model="gpt-4o",      max_score=70),   # medium complexity
    RoutingTier(model="o3-mini",     max_score=100),  # long / complex (catch-all)
])

kit = gpt.Chat(
    model="auto",            # ← triggers routing
    default_prompt=prompt,
    router=router,
)

# Short, simple question → score ~10 → routes to gpt-4o-mini
r1 = kit.chat(gpt.ChatConfig(user_message="What is 2+2?"))
print(r1.content)
# "4"
# Routed to: gpt-4o-mini   (cheapest tier)

# Long, technical question → score ~65 → routes to gpt-4o
r2 = kit.chat(gpt.ChatConfig(
    user_message=(
        "Explain the difference between RLHF, DPO, and PPO in the context of "
        "fine-tuning large language models for instruction following."
    )
))
print(r2.content)
# "RLHF (Reinforcement Learning from Human Feedback) is..."
# Routed to: gpt-4o

# Check which model was actually used
print(r2.raw.model)
# "gpt-4o"
```

**Works identically with Google and Anthropic kits:**

```python
from ractogateway import anthropic_developer_kit as claude
from ractogateway.routing import CostAwareRouter, RoutingTier

router = CostAwareRouter(tiers=[
    RoutingTier(model="claude-haiku-4-5-20251001", max_score=40),
    RoutingTier(model="claude-sonnet-4-6",         max_score=100),
])

kit = claude.Chat(model="auto", default_prompt=prompt, router=router)
```

**`RoutingTier` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `str` | required | Provider model identifier for this tier |
| `max_score` | `int` | `100` | Inclusive upper-bound complexity score (0–100). Last tier should always be `100` |

**Routing score algorithm:**

| Signal | Weight | Notes |
| --- | --- | --- |
| Token estimate | up to 60 pts | `len(text) // 4` tokens, saturates at 400 tokens |
| Keyword hits | up to 40 pts | Matches against 25 complexity keywords (e.g. `"analyze"`, `"compare"`, `"optimize"`) |

---

### Token Truncation

Automatically trims conversation history when it approaches the model's context limit. Uses a sliding-window strategy: always keeps the first `keep_first_n` messages (system context) and the last `keep_last_n` messages (recent context), dropping the middle.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.truncation import TokenTruncator, TruncationConfig

truncator = TokenTruncator(TruncationConfig(
    keep_first_n=2,       # always keep the 2 oldest history messages
    keep_last_n=8,        # always keep the 8 most recent messages
    safety_margin=512,    # reserve 512 tokens for the completion
))

kit = gpt.Chat(model="gpt-4o", default_prompt=prompt, truncator=truncator)

# Build a very long history (simulating a long conversation)
history = [
    gpt.Message(role=gpt.MessageRole.USER,      content=f"Question {i}")
    for i in range(200)
]

# The truncator silently trims history before sending to the API
response = kit.chat(gpt.ChatConfig(
    user_message="Summarize our conversation.",
    history=history,   # 200 messages — would overflow context without truncation
))
print(response.content)
# "Our conversation covered Questions 0 through 199..."
# History was automatically trimmed to fit within the context window.
```

**`TruncationConfig` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_context_tokens` | `int \| None` | `None` | Override context limit. `None` = auto-detect from model name |
| `keep_first_n` | `int` | `2` | History messages always kept from the start |
| `keep_last_n` | `int` | `6` | History messages always kept from the end |
| `safety_margin` | `int` | `512` | Token buffer reserved for the completion |
| `token_counter` | `Callable[[str], int] \| None` | `None` | Custom token counter. `None` = `len(text) // 4` approximation |

**Built-in context limits (auto-detected by model name):**

| Model | Context tokens |
| --- | --- |
| `gpt-4o`, `gpt-4o-*` | 128,000 |
| `gpt-4-turbo*` | 128,000 |
| `gpt-4` | 8,192 |
| `gpt-3.5-turbo` | 16,385 |
| `gemini-2.0-flash*` | 1,048,576 |
| `gemini-1.5-pro*` | 2,097,152 |
| `claude-*` (opus, sonnet, haiku) | 200,000 |

**Exact token counting with tiktoken (OpenAI models):**

```python
import tiktoken
from ractogateway.truncation import TokenTruncator, TruncationConfig

enc = tiktoken.encoding_for_model("gpt-4o")

truncator = TokenTruncator(TruncationConfig(
    token_counter=lambda text: len(enc.encode(text)),   # exact count
    keep_first_n=2,
    keep_last_n=10,
))
```

Install tiktoken: `pip install ractogateway[cache]`

---

### Batch Processing

Submit thousands of non-urgent requests using provider Batch APIs at approximately **50 % of standard API cost**. Jobs are processed asynchronously — you submit, poll for completion, then retrieve results.

#### OpenAI Batch Processor

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway import RactoPrompt
from ractogateway.batch import OpenAIBatchProcessor, BatchItem

prompt = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Answer the user's question briefly.",
    constraints=["Be concise."],
    tone="Friendly",
    output_format="text",
)

processor = OpenAIBatchProcessor(
    model="gpt-4o-mini",
    default_prompt=prompt,
)

# Build your batch
items = [
    BatchItem(custom_id="q1", user_message="What is the capital of France?"),
    BatchItem(custom_id="q2", user_message="What is 2 + 2?"),
    BatchItem(custom_id="q3", user_message="Explain Python decorators in one sentence."),
]

# Submit and block until complete (poll every 60 s, timeout 24 h)
results = processor.submit_and_wait(items, prompt=prompt)

for r in results:
    if r.ok:
        print(f"{r.custom_id}: {r.response.content}")
    else:
        print(f"{r.custom_id}: ERROR — {r.error}")

# Output:
# q1: The capital of France is Paris.
# q2: 4
# q3: A decorator is a function that wraps another function to extend its behavior without modifying it.
```

**Fine-grained control (submit → poll → fetch separately):**

```python
# 1. Submit — returns immediately
job = processor.submit_batch(items, prompt=prompt)
print(job.job_id)      # "batch_abc123"
print(job.status)      # BatchStatus.IN_PROGRESS
print(job.created_at)  # 1740000000.0  (Unix timestamp)

# 2. Poll until done
import time
while True:
    job = processor.poll_status(job.job_id)
    print(job.status)  # BatchStatus.IN_PROGRESS / FINALIZING / COMPLETED
    if job.status.value == "completed":
        break
    time.sleep(60)

# 3. Fetch results
results = processor.get_results(job.job_id)
```

**Async variant:**

```python
import asyncio

async def run():
    results = await processor.asubmit_and_wait(
        items,
        prompt=prompt,
        poll_interval_s=30.0,   # check every 30 s
    )
    for r in results:
        print(r.custom_id, r.response.content if r.ok else r.error)

asyncio.run(run())
```

#### Anthropic Batch Processor

```python
from ractogateway import anthropic_developer_kit as claude
from ractogateway.batch import AnthropicBatchProcessor, BatchItem

processor = AnthropicBatchProcessor(
    model="claude-haiku-4-5-20251001",   # cheapest Claude model
    default_prompt=prompt,
)

items = [
    BatchItem(custom_id="task1", user_message="Summarize quantum computing in 2 sentences."),
    BatchItem(custom_id="task2", user_message="List 3 benefits of exercise."),
]

results = processor.submit_and_wait(items)

for r in results:
    if r.ok:
        print(f"[{r.custom_id}] {r.response.content}")
    else:
        print(f"[{r.custom_id}] FAILED: {r.error}")

# Output:
# [task1] Quantum computing uses quantum mechanics principles like superposition and
#         entanglement to perform computations far beyond classical computers' reach.
#         It promises breakthroughs in cryptography, drug discovery, and optimization.
# [task2] 1. Improves cardiovascular health  2. Boosts mood  3. Increases energy levels
```

**`BatchItem` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `custom_id` | `str` | required | Your identifier for this request (returned in results) |
| `user_message` | `str` | required | The user turn content |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_tokens` | `int` | `4096` | Max completion tokens |
| `extra` | `dict` | `{}` | Provider-specific pass-through parameters |

**`BatchResult` fields:**

| Field | Type | Description |
| --- | --- | --- |
| `custom_id` | `str` | Your identifier from `BatchItem` |
| `response` | `LLMResponse \| None` | Parsed response (populated on success) |
| `error` | `str \| None` | Error message (populated on failure) |
| `ok` | `bool` | `True` if the request succeeded |
| `raw` | `Any` | Unmodified provider result object |

**`BatchStatus` values:**

| Value | Description |
| --- | --- |
| `PENDING` | Job created, not yet submitted |
| `IN_PROGRESS` | Provider is processing |
| `FINALIZING` | OpenAI is preparing results |
| `COMPLETED` | All results available |
| `FAILED` | Job failed |
| `EXPIRED` | Job expired before completion |
| `CANCELLING` | Cancellation in progress |
| `CANCELLED` | Job was cancelled |

---

### Combining All Optimizations

All four middleware features can be stacked on the same kit. The pipeline runs in this order on every `chat()` / `achat()` call:

```text
TokenTruncator → ExactMatchCache → SemanticCache → CostAwareRouter → API call → write caches
```

Cache hits short-circuit the pipeline — if an exact or semantic match is found, the API call and router are never invoked.

```python
import tiktoken
from ractogateway import openai_developer_kit as gpt, RactoPrompt
from ractogateway.cache import ExactMatchCache, SemanticCache
from ractogateway.routing import CostAwareRouter, RoutingTier
from ractogateway.truncation import TokenTruncator, TruncationConfig

# --- Embedding function (reuse your RAG embedder or any provider) ---
embed_kit = gpt.Chat(model="gpt-4o")

def embedder(text: str) -> list[float]:
    return embed_kit.embed(gpt.EmbeddingConfig(texts=[text])).vectors[0].vector

# --- Token counter (exact, via tiktoken) ---
enc = tiktoken.encoding_for_model("gpt-4o")

# --- Build the fully optimized kit ---
prompt = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Answer the user's question clearly and concisely.",
    constraints=["Never fabricate facts."],
    tone="Friendly",
    output_format="text",
)

kit = gpt.Chat(
    model="auto",                         # cost-aware routing enabled
    default_prompt=prompt,

    exact_cache=ExactMatchCache(          # identical-request cache
        max_size=2048,
        ttl_seconds=3600,                 # 1 h
    ),
    semantic_cache=SemanticCache(         # near-duplicate cache
        embedder=embedder,
        threshold=0.95,
        max_size=512,
        ttl_seconds=1800,                 # 30 min
    ),
    router=CostAwareRouter(tiers=[        # complexity-based model routing
        RoutingTier(model="gpt-4o-mini", max_score=30),
        RoutingTier(model="gpt-4o",      max_score=100),
    ]),
    truncator=TokenTruncator(TruncationConfig(
        token_counter=lambda t: len(enc.encode(t)),
        keep_first_n=2,
        keep_last_n=8,
        safety_margin=512,
    )),
)

# First call — cache miss, router picks cheapest model
r1 = kit.chat(gpt.ChatConfig(user_message="What is Python?"))
print(r1.content)
# "Python is a high-level, interpreted programming language..."
# → exact cache miss, semantic cache miss, routed to gpt-4o-mini

# Identical call — exact cache hit, 0 ms, $0.00
r2 = kit.chat(gpt.ChatConfig(user_message="What is Python?"))
print(r2.content)
# "Python is a high-level, interpreted programming language..."
# → exact cache HIT

# Semantically equivalent phrasing — semantic cache hit
r3 = kit.chat(gpt.ChatConfig(user_message="Can you explain what Python is?"))
print(r3.content)
# "Python is a high-level, interpreted programming language..."
# → semantic cache HIT (cosine sim ≥ 0.95)

# Print combined stats
print("Exact cache:", kit.exact_cache.stats)
# Exact cache: CacheStats(hits=1, misses=1, size=1, hit_rate=50.0%)

print("Semantic cache:", kit.semantic_cache.stats)
# Semantic cache: CacheStats(hits=1, misses=1, size=1, hit_rate=50.0%)
```

**Combined savings summary:**

| Scenario | Without optimization | With optimization |
| --- | --- | --- |
| 1,000 identical queries | 1,000 API calls | 1 API call + 999 cache hits |
| 1,000 semantically similar queries | 1,000 API calls | ~1–5 API calls + 995–999 cache hits |
| Mixed complexity (80 % simple) | 1,000 × expensive model | 800 × cheap model + 200 × expensive model |
| 10,000 non-urgent tasks | 10,000 standard calls | 10,000 batch calls (~50 % cost) |

---

```text
src/ractogateway/
├── __init__.py                          # Top-level: RactoPrompt, ToolRegistry, kits, RAG, fine-tuning
├── py.typed                             # PEP 561 typed package marker
│
├── _models/                             # Shared Pydantic input/output models
│   ├── chat.py                          #   ChatConfig, Message, MessageRole
│   ├── stream.py                        #   StreamChunk, StreamDelta
│   └── embedding.py                     #   EmbeddingConfig, EmbeddingResponse, EmbeddingVector
│
├── prompts/                             # RACTO Prompt Engine
│   └── engine.py                        #   RactoPrompt, RactoFile, compile(), to_messages()
│
├── finetune/                            # Multimodal Fine-Tuning Pipeline
│   ├── dataset.py                       #   RactoTrainingMessage, RactoTrainingExample, RactoDataset
│   ├── openai_tuner.py                  #   OpenAIFineTuner
│   ├── gemini_tuner.py                  #   GeminiFineTuner
│   └── anthropic_tuner.py               #   AnthropicFineTuner
│
├── tools/                               # Tool Registry
│   └── registry.py                      #   @tool decorator, ToolRegistry, ToolSchema, ParamSchema
│
├── gateway/                             # Low-Level Gateway
│   └── runner.py                        #   Gateway (wraps any BaseLLMAdapter)
│
├── adapters/                            # Internal provider adapters (Adapter Pattern)
│   ├── base.py                          #   BaseLLMAdapter ABC, LLMResponse, FinishReason, ToolCallResult
│   ├── openai_kit.py                    #   OpenAILLMKit
│   ├── google_kit.py                    #   GoogleLLMKit
│   └── anthropic_kit.py                 #   AnthropicLLMKit
│
├── openai_developer_kit/                # OpenAI Developer Kit (import as gpt)
│   └── kit.py                           #   OpenAIDeveloperKit (Chat alias)
│
├── google_developer_kit/                # Google Developer Kit (import as gemini)
│   └── kit.py                           #   GoogleDeveloperKit (Chat alias)
│
├── anthropic_developer_kit/             # Anthropic Developer Kit (import as claude)
│   └── kit.py                           #   AnthropicDeveloperKit (Chat alias)
│
├── redis/                               # Redis Infrastructure (pip install ractogateway[redis])
│   ├── _models.py                       #   RateLimitConfig, ChatMemoryConfig
│   ├── exact_cache.py                   #   RedisExactCache (drop-in for ExactMatchCache)
│   ├── rate_limiter.py                  #   RedisRateLimiter (fleet-wide token-bucket)
│   └── chat_memory.py                   #   RedisChatMemory (sliding-window conversation history)
│
├── celery/                              # Celery Task Queue (pip install ractogateway[celery])
│   ├── _models.py                       #   TaskStatus, TaskResult, RetryConfig
│   └── worker.py                        #   RactoCeleryWorker (generate, ingest_document, parallel_batch)
│
└── rag/                                 # RAG Pipeline
    ├── pipeline.py                      #   RactoRAG
    ├── _models/                         #   Document, Chunk, ChunkMetadata, RetrievalResult, RAGResponse
    ├── readers/                         #   TextReader, HtmlReader, PdfReader, WordReader, SpreadsheetReader, ImageReader, FileReaderRegistry
    ├── chunkers/                        #   FixedChunker, RecursiveChunker, SentenceChunker, SemanticChunker
    ├── processors/                      #   TextCleaner, Lemmatizer, ProcessingPipeline
    ├── embedders/                       #   OpenAIEmbedder, GoogleEmbedder, VoyageEmbedder
    └── stores/                          #   InMemoryVectorStore, ChromaStore, FAISSStore, Pinecone, Qdrant, Weaviate, Milvus, PGVector
```

### Design Principles

- **Lazy provider imports** — `openai`, `google-genai`, and `anthropic` SDKs are only imported when you instantiate a kit. `import ractogateway` never fails due to a missing optional dependency.
- **Pydantic everywhere** — Every input is a validated model. Every output is a typed model. No `dict[str, Any]` at the API boundary.
- **Composition over inheritance** — Developer kits compose internal adapters rather than extending them, keeping the public API clean.
- **Sync + async parity** — Every method has both a synchronous and asynchronous variant.
- **Provider-agnostic tool schemas** — Define tools once, use them with any provider. Internal adapters handle the translation.
- **Auto-JSON parsing** — Response content is automatically stripped of markdown code fences and JSON is parsed — no `json.loads()` needed.

---

## MCP (Model Context Protocol)

RactoGateway includes first-class MCP support for serving tools, consuming remote tools, and running automatic tool loops with OpenAI, Gemini, or Claude kits.

### MCP Components

| Component | What it does |
| --- | --- |
| `RactoMCPServer` | Exposes a `ToolRegistry` as an MCP server (`stdio` or `sse`). |
| `RactoMCPClient` | Connects to one MCP server and calls tools. |
| `MCPMultiClient` | Connects to multiple MCP servers and merges tools. |
| `MCPAgent` | Runs `LLM -> tool -> continue` loops automatically. |
| `MCPClientConfig` | Transport config (`stdio`, `sse`, `streamable-http`). |
| `MCPServerConfig` | Server metadata (`name`, `description`, `version`). |
| `MCPToolResult` | Normalized tool result (`content`, `is_error`). |

### 1) Build an MCP server (stdio)

```python
from ractogateway import ToolRegistry
from ractogateway.mcp import RactoMCPServer

registry = ToolRegistry()

@registry.register
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

server = RactoMCPServer.from_registry(registry, name="math-tools")
server.run(transport="stdio")  # blocks; for subprocess MCP clients
```

#### Input (Server Tool Call)

```json
{"tool": "add", "arguments": {"a": 7, "b": 5}}
```

#### Output (Server Tool Result)

```text
12
```

### 2) Connect and call a tool (sync one-shot)

```python
from ractogateway.mcp import MCPClientConfig, RactoMCPClient

config = MCPClientConfig(
    transport="stdio",
    command="python",
    args=["-m", "my_package.math_server"],
)

client = RactoMCPClient(config)
result = client.call_tool_sync("add", {"a": 20, "b": 22})

print(result.content)
print(result.is_error)
```

#### Output (Sync Client Call)

```text
42
False
```

### 3) Convert MCP tools to `ToolRegistry` and use with any kit

```python
import asyncio

from ractogateway import openai_developer_kit as gpt
from ractogateway.mcp import MCPClientConfig, RactoMCPClient

config = MCPClientConfig(
    transport="stdio",
    command="python",
    args=["-m", "my_package.weather_server"],
)

async def load_registry():
    async with RactoMCPClient(config) as client:
        return await client.to_registry()

registry = asyncio.run(load_registry())

kit = gpt.Chat(model="gpt-4o")
response = kit.chat(
    gpt.ChatConfig(
        user_message="What is weather in Tokyo?",
        tools=registry,
    )
)
print(response.content)
```

#### Input (Tool-Converted Registry)

```text
What is weather in Tokyo?
```

#### Output (Registry Chat Example)

```text
Tokyo weather is 26C and clear skies.
```

### 4) Merge multiple MCP servers

```python
import asyncio

from ractogateway.mcp import MCPClientConfig, MCPMultiClient

configs = [
    MCPClientConfig(transport="stdio", command="python", args=["-m", "pkg.math_server"]),
    MCPClientConfig(transport="sse", url="http://localhost:8001/sse"),
]

async def main() -> None:
    async with MCPMultiClient(configs) as multi:
        tools = await multi.list_tools()
        print([t.name for t in tools])

        result = await multi.call_tool("add", {"a": 2, "b": 3})
        print(result.content)

asyncio.run(main())
```

#### Output (Merged Servers Example)

```text
['add', 'search_docs', 'weather']
5
```

### 5) Run `MCPAgent` (automatic tool loop)

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.mcp import MCPAgent, MCPClientConfig

kit = gpt.Chat(model="gpt-4o")
configs = [
    MCPClientConfig(
        transport="stdio",
        command="python",
        args=["-m", "my_package.math_server"],
    )
]

agent = MCPAgent.from_mcp(kit, configs, max_turns=6)
response = agent.run(gpt.ChatConfig(user_message="What is 45 + 55?"))
print(response.content)
```

#### Input (MCPAgent Prompt)

```text
What is 45 + 55?
```

#### Output (MCPAgent Example)

```text
45 + 55 = 100.
```

### SSE server mode

If you want to host your MCP server over HTTP/SSE:

```python
from ractogateway import ToolRegistry
from ractogateway.mcp import RactoMCPServer

registry = ToolRegistry()

@registry.register
def ping() -> str:
    return "pong"

server = RactoMCPServer.from_registry(registry, name="network-tools")
server.run(transport="sse", host="0.0.0.0", port=8000)
```

SSE endpoint: `http://localhost:8000/sse`

### Important notes

- Use `pip install ractogateway[mcp]` for MCP core support.
- Use `pip install ractogateway[mcp-sse]` when running SSE server transport.
- Sync helpers (`*_sync`) should not be called inside a running event loop.

---

## Redis Infrastructure

Three production-ready utilities that replace or complement the built-in in-process modules when running across **multiple servers**. All three require only `pip install ractogateway[redis]` — no other configuration.

| Class | What it does | Replaces |
| --- | --- | --- |
| `RedisExactCache` | Distributed response cache — shared across every server in your fleet | `ExactMatchCache` (in-process only) |
| `RedisRateLimiter` | Fleet-wide token-budget rate limiting per user ID | Custom per-server solutions |
| `RedisChatMemory` | Sliding-window conversation history in a Redis List | In-memory `dict` approaches |

```bash
pip install ractogateway[redis]
```

---

### RedisExactCache — Distributed Response Cache

A **drop-in replacement** for `ExactMatchCache` with an identical public API. Swap it in wherever `ExactMatchCache` is accepted — including all developer-kit `exact_cache=` parameters — without changing any other code.

The cache is stored in Redis, so every server in your fleet reads from and writes to the same shared store. Responses cached by one replica are instantly available to all others.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.redis import RedisExactCache

cache = RedisExactCache(
    url="redis://localhost:6379/0",
    ttl_seconds=3600,    # entries expire after 1 hour
)

# Wire it in exactly like ExactMatchCache — nothing else changes
kit = gpt.Chat(model="gpt-4o", default_prompt=prompt, exact_cache=cache)

config = gpt.ChatConfig(user_message="What is the capital of France?")

r1 = kit.chat(config)
print(r1.content)
# "The capital of France is Paris."   ← Redis miss, API call made

r2 = kit.chat(config)
print(r2.content)
# "The capital of France is Paris."   ← Redis hit, $0.00, < 1 ms

# Works identically on a second server replica — cache is shared
stats = cache.stats
print(stats.hit_rate)   # 0.5
```

**`RedisExactCache` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `url` | `str` | `"redis://localhost:6379/0"` | Redis connection URL. Ignored when `client` is provided |
| `client` | `redis.Redis \| None` | `None` | Pre-built Redis client (useful for connection-pool sharing or mocking) |
| `ttl_seconds` | `float \| None` | `None` | Entry TTL passed to Redis `SET EX`. `None` = never expire |
| `key_prefix` | `str` | `"ractogateway:exact"` | Redis key namespace — change to avoid collisions between apps |

**Methods (identical to `ExactMatchCache`):**

| Method | Description |
| --- | --- |
| `get(user_message, system_prompt, model, temperature, max_tokens)` | Returns `LLMResponse` on hit, `None` on miss |
| `put(user_message, system_prompt, model, temperature, max_tokens, response)` | Store a response in Redis |
| `invalidate(...)` | Remove one specific entry. Returns `True` if it was present |
| `clear()` | Delete all entries matching the key prefix (uses `SCAN`, not `KEYS *`) |
| `stats` | Returns `CacheStats(hits, misses, size)` — hits/misses are in-memory counters |

---

### RedisRateLimiter — Fleet-Wide Rate Limiting

Enforces a **token budget per user ID** across every server in your fleet simultaneously. Uses a sliding 1-minute window via `INCRBY + EXPIRE` in a Redis pipeline — no Lua script, no race conditions that matter for rate limiting.

```python
from ractogateway.redis import RedisRateLimiter, RateLimitConfig

limiter = RedisRateLimiter(
    url="redis://localhost:6379/0",
    config=RateLimitConfig(max_tokens_per_minute=5_000),
)

# In your request handler — call this before every LLM call:
user_id = "user_42"
estimated_tokens = 800   # rough estimate of prompt + expected response

if not limiter.check_and_consume(user_id, tokens=estimated_tokens):
    raise RuntimeError("Rate limit exceeded — try again in a minute.")

response = kit.chat(gpt.ChatConfig(user_message=user_request))

# Check remaining budget (e.g. to return in response headers):
remaining = limiter.get_remaining(user_id)
print(f"Tokens remaining this minute: {remaining}")
# Tokens remaining this minute: 4200
```

**`RedisRateLimiter` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `url` | `str` | `"redis://localhost:6379/0"` | Redis connection URL |
| `client` | `redis.Redis \| None` | `None` | Pre-built Redis client |
| `config` | `RateLimitConfig \| None` | `None` | Limit config — defaults applied when `None` |

**`RateLimitConfig` fields:**

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_tokens_per_minute` | `int` | `10_000` | Maximum LLM tokens a single user may consume per minute |
| `key_prefix` | `str` | `"ractogateway:ratelimit"` | Redis key namespace |

**Methods:**

| Method | Returns | Description |
| --- | --- | --- |
| `check_and_consume(user_id, tokens=1)` | `bool` | `True` = request allowed (tokens consumed). `False` = budget exceeded (no tokens consumed) |
| `get_remaining(user_id)` | `int` | Remaining token budget for the current minute |
| `reset(user_id)` | `None` | Delete all rate-limit keys for a user (admin / testing) |

**Key format:** `"{key_prefix}:{user_id}:{unix_minute}"` — keys auto-expire after 60 seconds.

---

### RedisChatMemory — Sliding-Window Conversation History

Stores the last *N* message pairs per conversation in a Redis List. The history is shared across servers, survives rolling deployments, and is instantly accessible to both web servers and background workers.

```python
from ractogateway.redis import RedisChatMemory, ChatMemoryConfig

memory = RedisChatMemory(
    url="redis://localhost:6379/0",
    config=ChatMemoryConfig(
        max_turns=20,         # keep last 20 turns (40 messages)
        ttl_seconds=1800,     # conversations expire after 30 min of inactivity
    ),
)

conv_id = "conv_session_abc123"

# Append messages as the conversation progresses:
memory.append(conv_id, "user", "What is the capital of France?")
memory.append(conv_id, "assistant", "The capital of France is Paris.")
memory.append(conv_id, "user", "And what is its population?")

# Retrieve history to pass into the kit:
history = memory.get_history(conv_id)
# → [
#     {"role": "user",      "content": "What is the capital of France?"},
#     {"role": "assistant", "content": "The capital of France is Paris."},
#     {"role": "user",      "content": "And what is its population?"},
#   ]

print(memory.count(conv_id))   # 3

# Pass history into a ChatConfig:
response = kit.chat(gpt.ChatConfig(
    user_message="Compare it to Tokyo.",
    history=[gpt.Message(**m) for m in history],
))

# Clear when session ends:
memory.clear(conv_id)
```

**`RedisChatMemory` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `url` | `str` | `"redis://localhost:6379/0"` | Redis connection URL |
| `client` | `redis.Redis \| None` | `None` | Pre-built Redis client |
| `config` | `ChatMemoryConfig \| None` | `None` | Memory config — defaults applied when `None` |

**`ChatMemoryConfig` fields:**

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_turns` | `int` | `10` | Max conversation turns to keep. Stores up to `max_turns * 2` messages |
| `ttl_seconds` | `float \| None` | `None` | TTL refreshed on every `append()`. `None` = no expiry |
| `key_prefix` | `str` | `"ractogateway:memory"` | Redis key namespace — one List per `conversation_id` |

**Methods:**

| Method | Returns | Description |
| --- | --- | --- |
| `append(conversation_id, role, content)` | `None` | Add a message and trim to `max_turns * 2`. Refreshes TTL |
| `get_history(conversation_id)` | `list[dict[str, str]]` | All stored messages as `[{"role": ..., "content": ...}, ...]` |
| `clear(conversation_id)` | `None` | Delete the conversation from Redis |
| `count(conversation_id)` | `int` | Number of messages stored |

---

### Production Pattern — Combining Redis Utilities

All three utilities share the same Redis connection URL and complement each other. A typical production setup wires them together at the application level:

```python
from ractogateway import openai_developer_kit as gpt, RactoPrompt
from ractogateway.redis import (
    RedisExactCache,
    RedisRateLimiter,
    RedisChatMemory,
    RateLimitConfig,
    ChatMemoryConfig,
)

REDIS_URL = "redis://your-redis-host:6379/0"

# --- Shared infrastructure ---
cache = RedisExactCache(url=REDIS_URL, ttl_seconds=3600)
limiter = RedisRateLimiter(
    url=REDIS_URL,
    config=RateLimitConfig(max_tokens_per_minute=10_000),
)
memory = RedisChatMemory(
    url=REDIS_URL,
    config=ChatMemoryConfig(max_turns=20, ttl_seconds=1800),
)

# --- Kit with distributed cache ---
prompt = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Answer the user's question clearly.",
    constraints=["Never fabricate facts."],
    tone="Friendly",
    output_format="text",
)
kit = gpt.Chat(model="gpt-4o", default_prompt=prompt, exact_cache=cache)


# --- Request handler (e.g. FastAPI endpoint) ---
def handle_request(user_id: str, conv_id: str, user_message: str) -> str:
    # 1. Enforce rate limit before touching the LLM
    if not limiter.check_and_consume(user_id, tokens=500):
        raise RuntimeError(f"Rate limit exceeded. Remaining: {limiter.get_remaining(user_id)}")

    # 2. Load conversation history from Redis
    history = memory.get_history(conv_id)

    # 3. Call the kit (distributed cache checked automatically)
    response = kit.chat(gpt.ChatConfig(
        user_message=user_message,
        history=[gpt.Message(**m) for m in history],
    ))

    # 4. Persist the new turn back to Redis
    memory.append(conv_id, "user", user_message)
    memory.append(conv_id, "assistant", response.content or "")

    return response.content or ""
```

**What happens on every request:**

| Step | Action | Cost if cached |
| --- | --- | --- |
| Rate limit check | `INCRBY + EXPIRE` in Redis pipeline | < 1 ms |
| History load | `LRANGE` on Redis List | < 1 ms |
| Exact cache lookup | `GET` in Redis | < 1 ms — API call skipped entirely |
| LLM API call | Only if cache miss | Full cost + latency |
| History save | `RPUSH + LTRIM + EXPIRE` pipeline | < 1 ms |

---

## Celery Task Queue

`RactoCeleryWorker` is the background-task layer for long-running and retry-prone workflows.
It supports:

- Never-fail LLM generation with exponential-backoff retries.
- Background RAG ingestion (`read -> chunk -> embed -> store`) on worker nodes.
- Parallel fan-out inference with Celery `group()`.

Install:

```bash
pip install ractogateway[celery]
```

### Never-Fail LLM Generation

```python
# tasks.py (must be importable by BOTH app process and Celery workers)
from celery import Celery
from ractogateway import openai_developer_kit as gpt, RactoPrompt
from ractogateway.celery import RactoCeleryWorker, RetryConfig

prompt = RactoPrompt(
    role="You are a concise assistant.",
    aim="Answer clearly.",
    constraints=["Never fabricate facts."],
    tone="Professional",
    output_format="text",
)

celery_app = Celery(
    "ractogateway",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)
kit = gpt.Chat(model="gpt-4o", default_prompt=prompt)

worker = RactoCeleryWorker(
    celery_app,
    kit=kit,
    retry_config=RetryConfig(max_retries=3, initial_delay_s=2.0),
)

handle = worker.generate("Summarize this meeting transcript.")
result = worker.wait(handle.id, timeout_s=60.0)

print(result.status)
print(result.ok)
print(result.result["content"] if result.result else result.error)
```

#### Input (Generation Task)

```text
Summarize this meeting transcript.
```

#### Output (Generation Example)

```text
TaskStatus.SUCCESS
True
The meeting reviewed Q1 metrics and finalized two hiring decisions...
```

### Background Document Ingestion

```python
from celery import Celery
from ractogateway import openai_developer_kit as gpt, RactoRAG
from ractogateway.celery import RactoCeleryWorker
from ractogateway.rag.embedders import OpenAIEmbedder
from ractogateway.rag.stores import ChromaStore

celery_app = Celery(
    "ractogateway",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

kit = gpt.Chat(model="gpt-4o")
rag = RactoRAG(
    vector_store=ChromaStore(collection="docs", persist_directory="./db"),
    embedder=OpenAIEmbedder(model="text-embedding-3-large"),
    llm_kit=kit,
)

worker = RactoCeleryWorker(celery_app, kit=kit, rag=rag)

job = worker.ingest_document("./docs/policy.pdf", source="policy_v1")
status = worker.wait(job.id, timeout_s=180.0)

print(status.status)
print(len(status.result) if status.result else status.error)
```

#### Output (Ingestion Example)

```text
TaskStatus.SUCCESS
42
```

### Parallel Batch Inference

```python
from celery import Celery
from ractogateway import openai_developer_kit as gpt
from ractogateway.batch import BatchItem
from ractogateway.celery import RactoCeleryWorker

celery_app = Celery(
    "ractogateway",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)
kit = gpt.Chat(model="gpt-4o-mini")
worker = RactoCeleryWorker(celery_app, kit=kit)

group_result = worker.parallel_batch(
    [
        BatchItem(custom_id="q1", user_message="What is Python?"),
        BatchItem(custom_id="q2", user_message="What is Redis?"),
    ]
)

results = worker.wait_parallel(group_result, timeout_s=120.0)
for r in results:
    print(r.task_id, r.status, r.ok)
```

#### Output (Parallel Batch Example)

```text
e4c7... TaskStatus.SUCCESS True
7ad3... TaskStatus.SUCCESS True
```

### RetryConfig — Exponential Backoff Policy

`RetryConfig` controls transient-failure retries in Celery tasks.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `max_retries` | `int` | `3` | Retry attempts after first failure |
| `initial_delay_s` | `float` | `2.0` | Delay before first retry |
| `backoff_factor` | `float` | `2.0` | Delay multiplier per retry |
| `max_delay_s` | `float` | `300.0` | Upper bound for retry delay |

Delay formula used by worker tasks:

```text
delay = min(initial_delay_s * backoff_factor**attempt, max_delay_s)
```

With defaults: `2s -> 4s -> 8s` (then retries are exhausted).

### Worker Startup

Because Celery workers run in separate processes, the module that instantiates
`RactoCeleryWorker` must be imported by the worker process.

```bash
celery -A tasks.celery_app worker --loglevel=info
```

---

## Environment Variables

| Variable | Provider | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI | API key — used when `api_key` is not passed to the constructor |
| `GEMINI_API_KEY` | Google | API key — used when `api_key` is not passed to the constructor |
| `ANTHROPIC_API_KEY` | Anthropic | API key — used when `api_key` is not passed to the constructor |

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

```bash
# Clone and install in development mode
git clone https://github.com/IAMPathak2702/RactoGateway.git
cd RactoGateway
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2026 Ved Prakash Pathak

---

## Author

### Ved Prakash Pathak

- GitHub: [@IAMPathak2702](https://github.com/IAMPathak2702)
- Email: [vp.ved.vpp@gmail.com](mailto:vp.ved.vpp@gmail.com)
