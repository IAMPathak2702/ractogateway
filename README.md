# RactoGateway

**One Python package for all production-grade LLM solutions.**

RactoGateway is a unified AI SDK that gives you a single, clean interface to OpenAI, Google Gemini, and Anthropic Claude — with built-in anti-hallucination prompting, strict Pydantic validation, streaming, tool calling, embeddings, fine-tuning, and a full RAG pipeline. No more messy JSON dicts. No more provider lock-in. No more inconsistent response formats.

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/ractogateway/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-GitHub-green.svg)](https://github.com/IAMPathak2702/RactoGateway)

---

## Table of Contents

- [Why RactoGateway?](#why-ractogateway)
- [Installation](#installation)
- [RACTO Prompt Engine](#racto-prompt-engine)
- [Developer Kits](#developer-kits)
- [Streaming](#streaming)
- [Async Support](#async-support)
- [Embeddings](#embeddings)
- [Tool Calling](#tool-calling)
- [Validated Response Models](#validated-response-models)
- [Multi-turn Conversations](#multi-turn-conversations)
- [Multimodal Attachments — Images & Files](#multimodal-attachments)
- [Switching Providers](#switching-providers)
- [Fine-Tuning](#fine-tuning)
- [RAG — Retrieval-Augmented Generation](#rag)
- [Architecture](#architecture)
- [Environment Variables](#environment-variables)

---

## Why RactoGateway?

Every LLM provider has a different SDK, different request format, different response structure, and different tool-calling schema. Building production AI applications means writing glue code, parsing deeply nested objects, and manually stripping markdown fences from JSON responses.

RactoGateway solves this by providing:

- **RACTO Prompt Engine** — a structured prompt framework (Role, Aim, Constraints, Tone, Output) that compiles into optimized, anti-hallucination system prompts
- **Three Developer Kits** — `opd` (OpenAI), `god` (Google), `anth` (Anthropic) — each with `chat()`, `achat()`, `stream()`, `astream()`, `embed()`, and `aembed()`
- **Strict Pydantic models** for every input and output — no raw dicts anywhere
- **Automatic JSON parsing** — responses are cleaned of markdown fences and auto-parsed
- **Unified tool calling** — define tools once as Python functions, use them with any provider
- **Streaming with typed chunks** — every `StreamChunk` has `.delta.text`, `.accumulated_text`, `.is_final`, `.usage`
- **RAG pipeline** — ingest files, embed, store, retrieve, and generate answers with one class

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

# Development (all providers + testing + linting)
pip install ractogateway[dev]
```

**Requirements:** Python 3.10+, Pydantic 2.0+

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

### All `RactoPrompt` Fields

| Field | Type | Required | Default | Description |
| --- | --- | :---: | --- | --- |
| `role` | `str` | Yes | — | Who the model is |
| `aim` | `str` | Yes | — | Task objective |
| `constraints` | `list[str]` | Yes | — | Hard rules (min 1) |
| `tone` | `str` | Yes | — | Communication style |
| `output_format` | `str \| type[BaseModel]` | Yes | — | `"json"`, `"text"`, `"markdown"`, free-form, or a Pydantic class |
| `context` | `str \| None` | No | `None` | Domain background injected between AIM and CONSTRAINTS |
| `examples` | `list[dict]` | No | `None` | Few-shot pairs — each dict needs `"input"` and `"output"` keys |
| `anti_hallucination` | `bool` | No | `True` | Append `[GUARDRAILS]` block |

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

RactoGateway provides three developer kits — one per provider. Import them as short aliases:

```python
from ractogateway import openai_developer_kit as opd       # OpenAI / Azure OpenAI
from ractogateway import google_developer_kit as god        # Google Gemini
from ractogateway import anthropic_developer_kit as anth    # Anthropic Claude
```

### Kit Constructors

```python
# OpenAI
kit = opd.OpenAIDeveloperKit(
    model="gpt-4o",                            # required
    api_key="sk-...",                          # or set OPENAI_API_KEY env var
    base_url="https://custom-proxy.com/v1",    # optional (Azure, proxies)
    embedding_model="text-embedding-3-small",  # default
    default_prompt=prompt,                     # optional — used when ChatConfig.prompt is not set
)

# Google Gemini
kit = god.GoogleDeveloperKit(
    model="gemini-2.0-flash",                  # required
    api_key="AIza...",                         # or set GEMINI_API_KEY env var
    embedding_model="text-embedding-004",      # default
    default_prompt=prompt,                     # optional
)

# Anthropic Claude
kit = anth.AnthropicDeveloperKit(
    model="claude-sonnet-4-6",                 # required
    api_key="sk-ant-...",                      # or set ANTHROPIC_API_KEY env var
    default_prompt=prompt,                     # optional
)
```

### Method Reference

| Method | `opd` | `god` | `anth` | Description |
| --- | :---: | :---: | :---: | --- |
| `chat(config)` | Yes | Yes | Yes | Synchronous chat completion → `LLMResponse` |
| `achat(config)` | Yes | Yes | Yes | Async chat completion → `LLMResponse` |
| `stream(config)` | Yes | Yes | Yes | Sync streaming → yields `StreamChunk` |
| `astream(config)` | Yes | Yes | Yes | Async streaming → yields `StreamChunk` |
| `embed(config)` | Yes | Yes | — | Sync embeddings → `EmbeddingResponse` |
| `aembed(config)` | Yes | Yes | — | Async embeddings → `EmbeddingResponse` |

> Anthropic does not offer a native embedding API. Use the OpenAI or Google kit for embeddings.

---

### `ChatConfig` — Input Model

The single input object for `chat()`, `achat()`, `stream()`, and `astream()`.

```python
config = opd.ChatConfig(
    user_message="Explain monads in simple terms.",   # required
    prompt=prompt,                                     # optional — overrides kit default
    temperature=0.3,                                   # 0.0–2.0, default 0.0
    max_tokens=2048,                                   # default 4096
    tools=my_tool_registry,                            # optional ToolRegistry
    response_model=MyPydanticModel,                    # optional output validation
    history=[                                          # optional multi-turn context
        opd.Message(role=opd.MessageRole.USER, content="What is FP?"),
        opd.Message(role=opd.MessageRole.ASSISTANT, content="Functional programming is..."),
    ],
    attachments=[RactoFile.from_path("diagram.png")], # optional file attachments
    extra={"top_p": 0.9, "seed": 42},                 # provider-specific pass-through
)
```

---

### `LLMResponse` — Chat Output

Returned by `chat()` and `achat()`. Same shape for all three providers.

```python
response = kit.chat(opd.ChatConfig(user_message="What is 2 + 2?"))

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
kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt)
response = kit.chat(opd.ChatConfig(user_message="My name is Alice and I am 30 years old."))

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

---

## Streaming

`stream()` and `astream()` yield `StreamChunk` objects — one per streaming event.

```python
from ractogateway import openai_developer_kit as opd, RactoPrompt

prompt = RactoPrompt(
    role="You are a Python teacher.",
    aim="Explain the concept clearly.",
    constraints=["Use simple language.", "Give a short code example."],
    tone="Friendly",
    output_format="text",
)
kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt)

for chunk in kit.stream(opd.ChatConfig(user_message="Explain Python generators")):
    print(chunk.delta.text, end="", flush=True)   # incremental text
    if chunk.is_final:
        print()
        print(f"Finish reason : {chunk.finish_reason}")
        print(f"Tokens used   : {chunk.usage}")
        print(f"Full response : {chunk.accumulated_text[:80]}...")
```

**Example output:**

```
A generator in Python is a special function that yields values one at a time...

```python
def count_up(n):
    for i in range(n):
        yield i

for num in count_up(5):
    print(num)  # 0, 1, 2, 3, 4
```
Finish reason : FinishReason.STOP
Tokens used   : {"prompt_tokens": 55, "completion_tokens": 120, "total_tokens": 175}
Full response : A generator in Python is a special function that yields values one at a time...

### `StreamChunk` Field Reference

| Field | Type | Description |
| --- | --- | --- |
| `delta.text` | `str` | Incremental text added in this chunk |
| `accumulated_text` | `str` | Full text accumulated from all chunks so far |
| `is_final` | `bool` | `True` only on the very last chunk |
| `finish_reason` | `FinishReason \| None` | Set only on the final chunk |
| `tool_calls` | `list[ToolCallResult]` | Populated on the final chunk only (if tool calls occurred) |
| `usage` | `dict[str, int]` | Token usage — populated on the final chunk only |
| `raw` | `Any` | Raw provider streaming event |

---

## Async Support

Every method has a matching async variant.

```python
import asyncio
from ractogateway import openai_developer_kit as opd, RactoPrompt

prompt = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Answer the user's question.",
    constraints=["Be concise."],
    tone="Friendly",
    output_format="text",
)
kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt)

async def main():
    # Async chat
    response = await kit.achat(opd.ChatConfig(user_message="What is SOLID?"))
    print(response.content)
    # "SOLID is a set of five object-oriented design principles: Single Responsibility,
    #  Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion."

    # Async streaming
    async for chunk in kit.astream(opd.ChatConfig(user_message="Explain SOLID briefly")):
        print(chunk.delta.text, end="", flush=True)
        if chunk.is_final:
            print(f"\nDone. Tokens: {chunk.usage}")

asyncio.run(main())
```

---

## Embeddings

### `EmbeddingConfig` — Input

```python
config = opd.EmbeddingConfig(
    texts=["Hello world", "Goodbye world"],   # required — list of strings
    model="text-embedding-3-large",            # optional (overrides kit default)
    dimensions=512,                            # optional — for models that support truncation
)
```

### `EmbeddingResponse` — Output

```python
from ractogateway import openai_developer_kit as opd

kit = opd.OpenAIDeveloperKit(model="gpt-4o", embedding_model="text-embedding-3-small")

response = kit.embed(opd.EmbeddingConfig(texts=["cat", "dog", "automobile"]))

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
| `vectors` | `list[EmbeddingVector]` | One per input text |
| `model` | `str` | Model used |
| `usage` | `dict[str, int]` | Token usage |

**`EmbeddingVector` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `index` | `int` | Position in the input list |
| `text` | `str` | The original input text |
| `embedding` | `list[float]` | The dense vector |

---

## Tool Calling

Define tools as plain Python functions — never write nested JSON dicts by hand. RactoGateway translates them into the correct format for each provider.

### Register Tools with a Decorator

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

### Use Tools with Any Kit

```python
config = opd.ChatConfig(
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

### `ToolCallResult` Field Reference

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` | Provider-assigned call ID |
| `name` | `str` | Function name |
| `arguments` | `dict[str, Any]` | Parsed argument dict |

---

## Validated Response Models

Force the LLM output into a specific Pydantic shape. If the model doesn't produce valid JSON matching your model, you get a clear validation error — not silent garbage.

```python
from pydantic import BaseModel
from ractogateway import openai_developer_kit as opd, RactoPrompt

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

kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt)

config = opd.ChatConfig(
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
from ractogateway import openai_developer_kit as opd, RactoPrompt

prompt = RactoPrompt(
    role="You are a helpful coding assistant.",
    aim="Help the user write and debug Python code.",
    constraints=["Always provide runnable code examples.", "Explain errors clearly."],
    tone="Friendly and educational",
    output_format="text",
)
kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt)

# Turn 1
r1 = kit.chat(opd.ChatConfig(user_message="Write a function to reverse a string in Python."))
print(r1.content)
# "def reverse_string(s: str) -> str:\n    return s[::-1]"

# Turn 2 — pass history so the model remembers turn 1
r2 = kit.chat(opd.ChatConfig(
    user_message="Now make it handle None input gracefully.",
    history=[
        opd.Message(role=opd.MessageRole.USER, content="Write a function to reverse a string in Python."),
        opd.Message(role=opd.MessageRole.ASSISTANT, content=r1.content),
    ],
))
print(r2.content)
# "def reverse_string(s: str | None) -> str | None:\n    if s is None:\n        return None\n    return s[::-1]"
```

---

## Multimodal Attachments

`RactoFile` lets you attach images, PDFs, plain-text files, and any binary file to a prompt. The attachment is automatically re-encoded into the format expected by the target provider.

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

### OpenAI Vision — End-to-End Example

```python
from ractogateway import RactoPrompt, openai_developer_kit as opd
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
kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt)

config = opd.ChatConfig(
    user_message="What does this chart show?",
    attachments=[RactoFile.from_path("sales_q4.png")],
)
response = kit.chat(config)

print(response.content)
# "The bar chart shows Q4 2024 sales figures across four regions.
#  North America leads with $4.2M, followed by Europe at $3.1M.
#  Asia-Pacific grew 18% quarter-over-quarter to reach $2.8M."
```

### Anthropic — Image + PDF Together

```python
from ractogateway import RactoPrompt, anthropic_developer_kit as anth
from ractogateway.prompts.engine import RactoFile

prompt = RactoPrompt(
    role="You are a financial analyst.",
    aim="Summarise the key financial metrics from the attached report and diagram.",
    constraints=["Only extract facts present in the documents.", "Be concise."],
    tone="Professional",
    output_format="text",
)
kit = anth.AnthropicDeveloperKit(model="claude-sonnet-4-6", default_prompt=prompt)

config = anth.ChatConfig(
    user_message="Summarise the attached report and explain the chart.",
    attachments=[
        RactoFile.from_path("annual_report.pdf"),
        RactoFile.from_path("revenue_chart.png"),
    ],
)
response = kit.chat(config)
print(response.content)
# "The annual report shows revenue of $12.4M for FY2024, up 22% YoY.
#  The chart illustrates steady monthly growth from January ($820K) to December ($1.4M)."
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

## Switching Providers

Same `ChatConfig`, different kit. Zero code changes to your prompt or config.

```python
from ractogateway import openai_developer_kit as opd
from ractogateway import google_developer_kit as god
from ractogateway import anthropic_developer_kit as anth
from ractogateway import RactoPrompt

prompt = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Answer the user's question accurately.",
    constraints=["Be concise.", "Cite sources when possible."],
    tone="Friendly and professional",
    output_format="text",
)

config = opd.ChatConfig(user_message="What is quantum computing?")

# OpenAI
okit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt)
print(okit.chat(config).content)
# "Quantum computing uses quantum bits (qubits) that can exist in superposition,
#  enabling calculations that classical computers cannot do efficiently..."

# Google Gemini
gkit = god.GoogleDeveloperKit(model="gemini-2.0-flash", default_prompt=prompt)
print(gkit.chat(config).content)
# "Quantum computing harnesses the principles of quantum mechanics..."

# Anthropic Claude
akit = anth.AnthropicDeveloperKit(model="claude-sonnet-4-6", default_prompt=prompt)
print(akit.chat(config).content)
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
from ractogateway import openai_developer_kit as opd
kit = opd.OpenAIDeveloperKit(model=fine_tuned_model)
response = kit.chat(opd.ChatConfig(user_message="What is a generator?"))
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

from ractogateway import google_developer_kit as god
kit = god.GoogleDeveloperKit(model=tuned_model)
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

| Member | Description |
| --- | --- |
| `RactoDataset.from_pairs(pairs, system)` | Build from `(user, assistant)` text tuples |
| `RactoDataset.from_jsonl(path, provider)` | Load a previously exported JSONL file |
| `.add(example)` | Append one `RactoTrainingExample` |
| `.extend(examples)` | Append a list of examples |
| `.validate(provider)` | Returns `list[str]` of errors (empty = valid) |
| `.split(train_ratio, seed)` | Returns `(train_ds, val_ds)` |
| `.shuffle(seed)` | Returns a new shuffled dataset |
| `.export_jsonl(path, provider, overwrite)` | Write to `.jsonl` on disk |
| `.to_jsonl_string(provider)` | Return JSONL as a `str` (no I/O) |
| `.summary()` | Dict: `examples`, `total_messages`, `multimodal_examples`, … |

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

```
Document → Read → Chunk → Process → Embed → Store
                                              ↓
                              Query → Embed → Retrieve → Generate → Answer
```

### Installation

```bash
pip install ractogateway[rag-all]    # everything
# or pick what you need:
pip install ractogateway[rag]        # base readers + NLP
pip install ractogateway[rag-pdf]    # PDF
pip install ractogateway[rag-chroma] # ChromaDB
```

### Quickstart — 4 Lines

```python
from ractogateway import openai_developer_kit as opd
from ractogateway.rag.pipeline import RactoRAG
from ractogateway.rag.embedders import OpenAIEmbedder
from ractogateway.rag.stores import InMemoryVectorStore

kit = opd.OpenAIDeveloperKit(model="gpt-4o")
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

**`RetrievalResult` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `chunk` | `Chunk` | The retrieved text chunk |
| `score` | `float` | Similarity score (higher = more relevant) |
| `rank` | `int` | 1-based rank |

**`Chunk` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `chunk_id` | `str` | UUID of this chunk |
| `doc_id` | `str` | UUID of the parent document |
| `content` | `str` | Text content |
| `embedding` | `list[float] \| None` | Dense embedding vector |
| `metadata` | `ChunkMetadata` | Provenance info |

**`ChunkMetadata` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `source` | `str` | File path or URL |
| `page` | `int \| None` | Page number for PDFs |
| `chunk_index` | `int` | 0-based chunk position |
| `total_chunks` | `int` | Total chunks in the document |
| `start_char` | `int` | Character offset (start) |
| `end_char` | `int` | Character offset (end) |
| `doc_id` | `str` | Parent document UUID |
| `extra` | `dict` | Caller-supplied metadata |

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

**`RAGResponse` field reference:**

| Field | Type | Description |
| --- | --- | --- |
| `answer` | `LLMResponse` | The generated answer (same as a normal `chat()` response) |
| `sources` | `list[RetrievalResult]` | Chunks used as context |
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

# Sentence — natural boundaries
rag = RactoRAG(
    vector_store=InMemoryVectorStore(),
    embedder=OpenAIEmbedder(),
    chunker=SentenceChunker(max_sentences=5),
    llm_kit=kit,
)
```

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
```

### Full RAG Pipeline Example — Production Setup

```python
from ractogateway import openai_developer_kit as opd, RactoPrompt
from ractogateway.rag.pipeline import RactoRAG
from ractogateway.rag.embedders import OpenAIEmbedder
from ractogateway.rag.stores import ChromaStore
from ractogateway.rag.chunkers import RecursiveChunker
from ractogateway.rag.processors import TextCleaner

# 1. Build the kit
kit = opd.OpenAIDeveloperKit(model="gpt-4o")

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

| Method | Description |
| --- | --- |
| `ingest(path, **metadata)` | Read, chunk, embed, and store a file → `list[Chunk]` |
| `ingest_dir(directory, pattern, **metadata)` | Recursively ingest all supported files → `list[Chunk]` |
| `ingest_text(text, source, **metadata)` | Ingest raw text directly → `list[Chunk]` |
| `aingest(path, **metadata)` | Async variant of `ingest` |
| `aingest_dir(directory, pattern, **metadata)` | Async variant of `ingest_dir` |
| `aingest_text(text, source, **metadata)` | Async variant of `ingest_text` |
| `retrieve(query, top_k, filters)` | Embed query and return top-k chunks → `list[RetrievalResult]` |
| `aretrieve(query, top_k, filters)` | Async variant of `retrieve` |
| `query(question, top_k, filters, prompt, temperature, max_tokens)` | Retrieve + generate → `RAGResponse` |
| `aquery(...)` | Async variant of `query` |
| `count()` | Total indexed chunks → `int` |
| `clear()` | Remove all indexed chunks |
| `store` | Access the underlying `BaseVectorStore` |
| `embedder` | Access the underlying `BaseEmbedder` |

---

## Architecture

```text
src/ractogateway/
├── __init__.py                          # Top-level: RactoPrompt, ToolRegistry, kits
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
│   └── registry.py                      #   @tool decorator, ToolRegistry, ToolSchema
│
├── adapters/                            # Internal provider adapters (Adapter Pattern)
│   ├── base.py                          #   BaseLLMAdapter ABC, LLMResponse, FinishReason
│   ├── openai_kit.py                    #   OpenAILLMKit
│   ├── google_kit.py                    #   GoogleLLMKit
│   └── anthropic_kit.py                 #   AnthropicLLMKit
│
├── openai_developer_kit/                # OpenAI Developer Kit (import as opd)
│   └── kit.py                           #   OpenAIDeveloperKit
│
├── google_developer_kit/                # Google Developer Kit (import as god)
│   └── kit.py                           #   GoogleDeveloperKit
│
├── anthropic_developer_kit/             # Anthropic Developer Kit (import as anth)
│   └── kit.py                           #   AnthropicDeveloperKit
│
└── rag/                                 # RAG Pipeline
    ├── pipeline.py                      #   RactoRAG
    ├── _models/                         #   Document, Chunk, ChunkMetadata, RetrievalResult, RAGResponse
    ├── readers/                         #   TextReader, PdfReader, WordReader, SpreadsheetReader, ImageReader, HtmlReader
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
