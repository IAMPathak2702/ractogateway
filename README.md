# RactoGateway

**One Python package for all production-grade LLM solutions.**

RactoGateway is a unified AI SDK that gives you a single, clean interface to OpenAI, Google Gemini, and Anthropic Claude — with built-in anti-hallucination prompting, strict Pydantic validation, streaming, tool calling, and embeddings. No more messy JSON dicts. No more provider lock-in. No more inconsistent response formats.

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/ractogateway/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-GitHub-green.svg)](https://github.com/IAMPathak2702/RactoGateway)

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

---

## Installation

```bash
# Core package (includes RACTO prompt engine and tool registry)
pip install ractogateway

# With a specific provider
pip install ractogateway[openai]
pip install ractogateway[google]
pip install ractogateway[anthropic]

# All providers
pip install ractogateway[all]

# Development (all providers + testing + linting)
pip install ractogateway[dev]
```

**Requirements:** Python 3.10+, Pydantic 2.0+

---

## Quick Start

### 1. Define a RACTO Prompt

Every prompt is a validated Pydantic model with five required fields:

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

### 2. Use a Developer Kit

```python
from ractogateway import openai_developer_kit as opd

kit = opd.OpenAIDeveloperKit(
    model="gpt-4o",
    api_key="sk-...",          # or set OPENAI_API_KEY env var
    default_prompt=prompt,
)

# Synchronous chat
response = kit.chat(opd.ChatConfig(user_message="Review this function:\ndef add(a, b): return a + b"))
print(response.content)        # cleaned text
print(response.parsed)         # auto-parsed JSON dict (if response was JSON)
print(response.usage)          # {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60}
```

### 3. Stream Responses

```python
for chunk in kit.stream(opd.ChatConfig(user_message="Explain Python generators")):
    print(chunk.delta.text, end="", flush=True)
    if chunk.is_final:
        print(f"\n\nTokens used: {chunk.usage}")
```

### 4. Async Support

```python
import asyncio

async def main():
    response = await kit.achat(opd.ChatConfig(user_message="What is SOLID?"))
    print(response.content)

    async for chunk in kit.astream(opd.ChatConfig(user_message="Explain SOLID")):
        print(chunk.delta.text, end="", flush=True)

asyncio.run(main())
```

---

## Developer Kits

RactoGateway provides three developer kits — one per provider. Each is a self-contained module with the kit class, all input models, and all output models.

```python
from ractogateway import openai_developer_kit as opd       # OpenAI / Azure OpenAI
from ractogateway import google_developer_kit as god        # Google Gemini
from ractogateway import anthropic_developer_kit as anth    # Anthropic Claude
```

### Method Reference

| Method | `opd` | `god` | `anth` | Description |
| --- | :---: | :---: | :---: | --- |
| `chat(config)` | Yes | Yes | Yes | Synchronous chat completion |
| `achat(config)` | Yes | Yes | Yes | Async chat completion |
| `stream(config)` | Yes | Yes | Yes | Sync streaming (yields `StreamChunk`) |
| `astream(config)` | Yes | Yes | Yes | Async streaming (yields `StreamChunk`) |
| `embed(config)` | Yes | Yes | -- | Sync embeddings |
| `aembed(config)` | Yes | Yes | -- | Async embeddings |

> Anthropic does not offer a native embedding API. Use the OpenAI or Google kit for embeddings.

### Kit Constructors

```python
# OpenAI
kit = opd.OpenAIDeveloperKit(
    model="gpt-4o",                            # required
    api_key="sk-...",                          # or OPENAI_API_KEY env var
    base_url="https://custom-proxy.com/v1",    # optional (Azure, proxies)
    embedding_model="text-embedding-3-small",  # default
    default_prompt=prompt,                     # optional
)

# Google Gemini
kit = god.GoogleDeveloperKit(
    model="gemini-2.0-flash",                  # required
    api_key="AIza...",                         # or GEMINI_API_KEY env var
    embedding_model="text-embedding-004",      # default
    default_prompt=prompt,                     # optional
)

# Anthropic Claude
kit = anth.AnthropicDeveloperKit(
    model="claude-sonnet-4-5-20250929",        # required
    api_key="sk-ant-...",                      # or ANTHROPIC_API_KEY env var
    default_prompt=prompt,                     # optional
)
```

---

## Input Models

All inputs are strictly validated Pydantic models. No raw dicts. No positional argument sprawl.

### `ChatConfig`

The single input for `chat()`, `achat()`, `stream()`, and `astream()`.

```python
config = opd.ChatConfig(
    user_message="Explain monads in simple terms.",   # required, min 1 char
    prompt=prompt,                                     # optional (falls back to kit default)
    temperature=0.3,                                   # 0.0–2.0, default 0.0
    max_tokens=2048,                                   # default 4096
    tools=my_tool_registry,                            # optional ToolRegistry
    response_model=MyPydanticModel,                    # optional output validation
    history=[                                          # optional multi-turn context
        opd.Message(role=opd.MessageRole.USER, content="What is FP?"),
        opd.Message(role=opd.MessageRole.ASSISTANT, content="Functional programming is..."),
    ],
    extra={"top_p": 0.9, "seed": 42},                 # provider-specific pass-through
)
```

### `EmbeddingConfig`

The input for `embed()` and `aembed()`.

```python
config = opd.EmbeddingConfig(
    texts=["Hello world", "Goodbye world"],   # required, min 1 text
    model="text-embedding-3-large",            # optional (overrides kit default)
    dimensions=512,                            # optional (for models that support it)
)
```

---

## Output Models

### `LLMResponse`

Returned by `chat()` and `achat()`. Unified across all providers.

| Field | Type | Description |
| --- | --- | --- |
| `content` | `str \| None` | Cleaned text (markdown fences stripped) |
| `parsed` | `dict \| list \| None` | Auto-parsed JSON (if response was valid JSON) |
| `tool_calls` | `list[ToolCallResult]` | Tool calls requested by the model |
| `finish_reason` | `FinishReason` | `STOP`, `TOOL_CALL`, `LENGTH`, `CONTENT_FILTER`, `ERROR` |
| `usage` | `dict[str, int]` | `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `raw` | `Any` | The unmodified provider response (escape hatch) |

### `StreamChunk`

Yielded by `stream()` and `astream()`. One per streaming event.

| Field | Type | Description |
| --- | --- | --- |
| `delta.text` | `str` | Incremental text for this chunk |
| `accumulated_text` | `str` | Full text accumulated so far |
| `is_final` | `bool` | `True` only on the last chunk |
| `finish_reason` | `FinishReason \| None` | Set on final chunk only |
| `tool_calls` | `list[ToolCallResult]` | Populated on final chunk only |
| `usage` | `dict[str, int]` | Populated on final chunk only |
| `raw` | `Any` | Raw provider streaming event |

### `EmbeddingResponse`

Returned by `embed()` and `aembed()`.

| Field | Type | Description |
| --- | --- | --- |
| `vectors` | `list[EmbeddingVector]` | Each has `.index`, `.text`, `.embedding` |
| `model` | `str` | Model used for embedding |
| `usage` | `dict[str, int]` | Token usage |

---

## RACTO Prompt Engine

The RACTO principle structures every prompt into five unambiguous sections:

| Letter | Field | Purpose |
| :---: | --- | --- |
| **R** | `role` | Who the model is |
| **A** | `aim` | What it must accomplish |
| **C** | `constraints` | Hard rules it must never violate |
| **T** | `tone` | Communication style |
| **O** | `output_format` | Exact shape of the response |

### Compiled Output

`prompt.compile()` produces a clearly delimited system prompt:

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

### Advanced: Pydantic Schema as Output Format

Pass a Pydantic model as `output_format` and the full JSON Schema is embedded directly in the prompt:

```python
from pydantic import BaseModel

class CodeReview(BaseModel):
    issues: list[str]
    severity: str
    suggestion: str

prompt = RactoPrompt(
    role="You are a code reviewer.",
    aim="Review the code.",
    constraints=["Only report real issues."],
    tone="Concise",
    output_format=CodeReview,     # JSON Schema embedded in prompt
)
```

### Optional Fields

| Field | Type | Description |
| --- | --- | --- |
| `context` | `str` | Domain background injected between AIM and CONSTRAINTS |
| `examples` | `list[dict]` | Few-shot input/output pairs for steering |
| `anti_hallucination` | `bool` | Append `[GUARDRAILS]` block (default `True`) |

---

## Multimodal Attachments — Images & Files

`RactoFile` lets you attach images, PDFs, plain-text files, and any other binary file to a `to_messages()` call.
The attachment is **automatically re-encoded** into the content-block schema expected by the target provider — you never write raw `image_url`, `inline_data`, or `source` dicts by hand.

### Creating a `RactoFile`

#### From a file path — MIME type is auto-detected

```python
from ractogateway.prompts.engine import RactoFile

img = RactoFile.from_path("/path/to/photo.jpg")       # image/jpeg
doc = RactoFile.from_path("/path/to/report.pdf")      # application/pdf
txt = RactoFile.from_path("/path/to/notes.txt")       # text/plain
```

#### From raw bytes — MIME type supplied explicitly

```python
# From an open file handle
with open("chart.png", "rb") as fh:
    img = RactoFile.from_bytes(fh.read(), "image/png", name="chart.png")

# From bytes already in memory (e.g. downloaded with requests)
import requests
resp = requests.get("https://example.com/diagram.png")
img = RactoFile.from_bytes(resp.content, "image/png", name="diagram.png")
```

### Passing Attachments to `to_messages()`

```python
messages = prompt.to_messages(
    "What does this chart show?",
    attachments=[img],          # list[RactoFile], any length
    provider="openai",          # "openai" | "anthropic" | "google" | "generic"
)
```

You can mix multiple files of different types in the same call:

```python
messages = prompt.to_messages(
    "Summarise the attached report and explain the diagram.",
    attachments=[
        RactoFile.from_path("report.pdf"),
        RactoFile.from_path("diagram.png"),
    ],
    provider="anthropic",
)
```

### Provider Content-Block Output

Each provider receives a different content-block shape. `to_messages()` handles the translation transparently.

#### OpenAI / Generic

Images and binary files become **`image_url`** blocks with an inline `data:` URI.
Text files become **`text`** blocks.

```python
# prompt.to_messages("Describe the image.", attachments=[jpeg_file], provider="openai")
[
    {"role": "system", "content": "<compiled RACTO system prompt>"},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB..."
                }
            },
            {"type": "text", "text": "Describe the image."}
        ]
    }
]
```

#### Anthropic Claude

- Images → **`image`** content block with `base64` source
- PDFs → **`document`** content block with `base64` source
- Text files → **`text`** content block (decoded UTF-8)
- Other binary → **`text`** block with a labelled base-64 payload

```python
# prompt.to_messages("Summarise.", attachments=[pdf_file], provider="anthropic")
[
    {"role": "system", "content": "<compiled RACTO system prompt>"},
    {
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjQK..."
                }
            },
            {"type": "text", "text": "Summarise."}
        ]
    }
]
```

#### Google Gemini

- Text files → **`text`** parts (decoded UTF-8)
- All other files → **`inline_data`** parts with `mime_type` and base-64 `data`

```python
# prompt.to_messages("What is in this image?", attachments=[png_file], provider="google")
[
    {"role": "system", "content": "<compiled RACTO system prompt>"},
    {
        "role": "user",
        "content": [
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": "iVBORw0KGgoAAAANS..."
                }
            },
            {"text": "What is in this image?"}
        ]
    }
]
```

### Supported File Types

| File type | MIME type | OpenAI | Anthropic | Google |
| --- | --- | :---: | :---: | :---: |
| JPEG image | `image/jpeg` | `image_url` block | `image` block | `inline_data` part |
| PNG image | `image/png` | `image_url` block | `image` block | `inline_data` part |
| GIF image | `image/gif` | `image_url` block | `image` block | `inline_data` part |
| WebP image | `image/webp` | `image_url` block | `image` block | `inline_data` part |
| PDF document | `application/pdf` | `image_url` block | `document` block | `inline_data` part |
| Plain text | `text/plain` | `text` block | `text` block | `text` part |
| Any other | `*/*` | `image_url` block (data URI) | labelled `text` block | `inline_data` part |

> MIME type detection is fully automatic when using `RactoFile.from_path()`. When using `RactoFile.from_bytes()`, supply the MIME type explicitly.

### Full End-to-End Example — OpenAI Vision

```python
from ractogateway import RactoPrompt
from ractogateway.prompts.engine import RactoFile
from ractogateway import openai_developer_kit as opd

prompt = RactoPrompt(
    role="You are a data analyst specialising in chart interpretation.",
    aim="Describe what the attached chart shows and extract the key insights.",
    constraints=[
        "Only describe what is visible in the image.",
        "Never invent data points that are not in the chart.",
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
# "The bar chart shows Q4 2024 sales figures across four regions..."
```

### Full End-to-End Example — Anthropic (Image + PDF)

```python
from ractogateway import RactoPrompt
from ractogateway.prompts.engine import RactoFile
from ractogateway import anthropic_developer_kit as anth

prompt = RactoPrompt(
    role="You are a financial analyst.",
    aim="Summarise the key financial metrics from the attached report and diagram.",
    constraints=["Only extract facts present in the documents.", "Be concise."],
    tone="Professional",
    output_format="text",
)

kit = anth.AnthropicDeveloperKit(model="claude-sonnet-4-5-20250929", default_prompt=prompt)

config = anth.ChatConfig(
    user_message="Summarise the attached report and explain the chart.",
    attachments=[
        RactoFile.from_path("annual_report.pdf"),
        RactoFile.from_path("revenue_chart.png"),
    ],
)

response = kit.chat(config)
print(response.content)
```

### Full End-to-End Example — From Bytes (e.g. API download)

```python
import requests
from ractogateway.prompts.engine import RactoFile

# Fetch an image from an external URL
resp = requests.get("https://example.com/chart.png")
chart = RactoFile.from_bytes(resp.content, "image/png", name="chart.png")

# Or read from a file handle
with open("report.pdf", "rb") as fh:
    pdf = RactoFile.from_bytes(fh.read(), "application/pdf", name="report.pdf")

messages = prompt.to_messages(
    "Analyse this chart and report.",
    attachments=[chart, pdf],
    provider="anthropic",
)
```

### `RactoFile` API Reference

| Member | Description |
| --- | --- |
| `RactoFile.from_path(path)` | Load from disk; MIME type auto-detected via `mimetypes`. Raises `FileNotFoundError` if path is missing. |
| `RactoFile.from_bytes(data, mime_type, name="")` | Wrap raw bytes; MIME type must be supplied explicitly. |
| `.data` | `bytes` — raw file content |
| `.mime_type` | `str` — MIME type string, e.g. `"image/png"` |
| `.name` | `str` — filename hint (empty string if not provided) |
| `.base64_data` | `str` — file bytes encoded as a base-64 ASCII string |
| `.is_image` | `bool` — `True` for `image/jpeg`, `image/png`, `image/gif`, `image/webp` |
| `.is_pdf` | `bool` — `True` for `application/pdf` |
| `.is_text` | `bool` — `True` for any `text/*` MIME type |

---

## Tool Calling

Define tools as Python functions — never write nested JSON dicts.

### Register Tools

```python
from ractogateway import ToolRegistry

registry = ToolRegistry()

@registry.register
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    :param city: The city name
    :param unit: Temperature unit (celsius or fahrenheit)
    """
    # Your implementation here
    return f"Weather in {city}: 22°{unit[0].upper()}"
```

### Use with Any Kit

```python
config = opd.ChatConfig(
    user_message="What's the weather in Tokyo?",
    tools=registry,
)
response = kit.chat(config)

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Call: {tc.name}({tc.arguments})")
        # Execute the function
        fn = registry.get_callable(tc.name)
        result = fn(**tc.arguments)
```

### Register Pydantic Models as Tools

```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    """Search the knowledge base."""
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum results to return")

registry.register(SearchQuery)
```

---

## Validated Response Models

Force the LLM output through a Pydantic model for guaranteed structure:

```python
class SentimentResult(BaseModel):
    sentiment: str       # "positive", "negative", "neutral"
    confidence: float    # 0.0 to 1.0
    reasoning: str

config = opd.ChatConfig(
    user_message="Analyze sentiment: 'This product is amazing!'",
    response_model=SentimentResult,
)
response = kit.chat(config)
print(response.parsed)
# {"sentiment": "positive", "confidence": 0.95, "reasoning": "Strong positive adjective 'amazing'"}
```

---

## Switching Providers

Same `ChatConfig`, different kit. That's it.

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

# Google Gemini
gkit = god.GoogleDeveloperKit(model="gemini-2.0-flash", default_prompt=prompt)
print(gkit.chat(config).content)

# Anthropic Claude
akit = anth.AnthropicDeveloperKit(model="claude-sonnet-4-5-20250929", default_prompt=prompt)
print(akit.chat(config).content)
```

---

## Fine-Tuning — Multimodal Training Pipeline

RactoGateway ships a production-grade fine-tuning module that works with **OpenAI**, **Google Gemini**, and **Anthropic Claude** using a single, unified dataset API.

```python
from ractogateway import (
    RactoDataset,
    RactoTrainingExample,
    RactoTrainingMessage,
    OpenAIFineTuner,
    GeminiFineTuner,
    AnthropicFineTuner,
)
# or via the sub-package
from ractogateway.finetune import RactoDataset, OpenAIFineTuner
```

### Core Classes

| Class | Role |
| --- | --- |
| `RactoTrainingMessage` | One conversation turn — role + text + optional `RactoFile` attachments |
| `RactoTrainingExample` | Full conversation (one training record) — list of `RactoTrainingMessage` |
| `RactoDataset` | Collection of examples with validation, split, and JSONL export |
| `OpenAIFineTuner` | Upload → train → poll on OpenAI |
| `GeminiFineTuner` | Create tuning job → poll on Google AI |
| `AnthropicFineTuner` | Upload → train → poll on Anthropic |

---

### Step 1 — Assemble Training Data

#### Text-only dataset (quickest path)

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
from ractogateway import RactoTrainingExample, RactoTrainingMessage, RactoDataset

example = RactoTrainingExample.from_conversation([
    ("system",    "You are a helpful travel assistant."),
    ("user",      "I want to visit Japan. What season is best?"),
    ("assistant", "Spring (March–May) for cherry blossoms, or Autumn (Sept–Nov) for foliage."),
    ("user",      "Which cities should I visit?"),
    ("assistant", "Tokyo, Kyoto, Osaka, and Hiroshima are the most popular."),
])

ds = RactoDataset([example])
```

#### Multimodal example (image + text)

```python
from ractogateway import RactoTrainingExample, RactoDataset
from ractogateway.prompts.engine import RactoFile

# From a file on disk
chart = RactoFile.from_path("sales_chart.png")

# From raw bytes (e.g. captured from an API or camera)
with open("invoice.png", "rb") as fh:
    invoice = RactoFile.from_bytes(fh.read(), "image/png", name="invoice.png")

example = RactoTrainingExample.from_pair(
    user="Describe the trend shown in this chart.",
    assistant="Revenue grew by 23% quarter-over-quarter, peaking in December.",
    system="You are a data analyst. Be concise and factual.",
    user_attachments=[chart],
)

ds = RactoDataset([example])
```

#### Add examples incrementally

```python
ds = RactoDataset()

ds.add(RactoTrainingExample.from_pair("Q1", "A1", system="You are helpful."))
ds.add(RactoTrainingExample.from_pair("Q2", "A2", system="You are helpful."))

# Or batch-extend
ds.extend([
    RactoTrainingExample.from_pair(u, a)
    for u, a in [("Q3", "A3"), ("Q4", "A4")]
])
```

---

### Step 2 — Validate and Split

```python
# Validate before uploading (catches empty content, wrong role order, etc.)
errors = ds.validate(provider="openai")   # or "anthropic" / "gemini"
if errors:
    for e in errors:
        print(e)
else:
    print("Dataset is valid.")

# Reproducible 80/20 train-validation split
train_ds, val_ds = ds.split(train_ratio=0.8, seed=42)
print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")
```

---

### Step 3 — Export to JSONL (optional inspection)

```python
train_ds.export_jsonl("train.jsonl",      provider="openai",     overwrite=True)
val_ds.export_jsonl("val.jsonl",          provider="openai",     overwrite=True)
train_ds.export_jsonl("train_ant.jsonl",  provider="anthropic",  overwrite=True)
train_ds.export_jsonl("train_gem.jsonl",  provider="gemini",     overwrite=True)
```

**OpenAI JSONL output** (`train.jsonl`):

```json
{"messages": [{"role": "system", "content": "You are a Python tutor."}, {"role": "user", "content": "What is a list?"}, {"role": "assistant", "content": "An ordered, mutable sequence."}]}
{"messages": [{"role": "system", "content": "You are a Python tutor."}, {"role": "user", "content": "What is a dict?"}, {"role": "assistant", "content": "A key-value mapping."}]}
```

**Anthropic JSONL output** (`train_ant.jsonl`):

```json
{"system": "You are a Python tutor.", "messages": [{"role": "user", "content": "What is a list?"}, {"role": "assistant", "content": "An ordered, mutable sequence."}]}
```

**Gemini JSONL output** (`train_gem.jsonl`):

```json
{"text_input": "What is a list?", "output": "An ordered, mutable sequence."}
```

**OpenAI multimodal JSONL output** (when the user turn has an image):

```json
{
  "messages": [
    {"role": "system", "content": "You are a data analyst."},
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR…"}},
        {"type": "text", "text": "Describe the trend."}
      ]
    },
    {"role": "assistant", "content": "Revenue grew 23% quarter-over-quarter."}
  ]
}
```

**Anthropic multimodal JSONL output**:

```json
{
  "system": "You are a data analyst.",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR…"}},
        {"type": "text", "text": "Describe the trend."}
      ]
    },
    {"role": "assistant", "content": "Revenue grew 23% quarter-over-quarter."}
  ]
}
```

---

### Step 4 — Fine-Tune

#### OpenAI — one call

```python
from ractogateway import OpenAIFineTuner

tuner = OpenAIFineTuner(api_key="sk-...")   # or set OPENAI_API_KEY

fine_tuned_model = tuner.run_pipeline(
    train_ds,
    model="gpt-4o-mini-2024-07-18",       # supports vision fine-tuning: gpt-4o-2024-08-06
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
```

#### OpenAI — step by step

```python
tuner = OpenAIFineTuner()

# 1. Upload files
train_file_id = tuner.upload_dataset(train_ds)
val_file_id   = tuner.upload_dataset(val_ds)

# 2. Create job
job_id = tuner.create_job(
    train_file_id,
    model="gpt-4o-mini-2024-07-18",
    validation_file=val_file_id,
    n_epochs=3,
    suffix="python-tutor",
)

# 3. Check status (non-blocking)
print(tuner.get_status(job_id))
# {"id": "ftjob-…", "status": "running", "model": "gpt-4o-mini-2024-07-18", …}

# 4. Stream training events
for event in tuner.list_events(job_id, limit=10):
    print(event["message"])

# 5. Block until done
fine_tuned_model = tuner.wait_for_completion(job_id, poll_interval=30)
```

#### Google Gemini — one call

```python
from ractogateway import GeminiFineTuner

tuner = GeminiFineTuner(api_key="AIza...")   # or set GEMINI_API_KEY

# Gemini requires single-turn text-pair examples
# (multimodal / multi-turn requires Vertex AI)
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

#### Google Gemini — step by step

```python
tuner = GeminiFineTuner()

operation = tuner.create_job(
    train_ds,
    base_model="models/gemini-1.5-flash-001-tuning",
    display_name="my-tuned-model",
    epoch_count=5,
)

# List / inspect / delete tuned models
for m in tuner.list_models():
    print(m["name"], m["state"])

tuned_model = tuner.wait_for_completion(operation)
```

#### Anthropic Claude — one call

```python
from ractogateway import AnthropicFineTuner

tuner = AnthropicFineTuner(api_key="sk-ant-...")   # or set ANTHROPIC_API_KEY

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

from ractogateway import anthropic_developer_kit as anth
kit = anth.AnthropicDeveloperKit(model=fine_tuned_model)
```

---

### Full Multimodal Pipeline Example

End-to-end: build a vision model that describes product images.

```python
import os
from ractogateway import (
    RactoDataset,
    RactoTrainingExample,
    OpenAIFineTuner,
    openai_developer_kit as opd,
)
from ractogateway.prompts.engine import RactoFile, RactoPrompt

# --- 1. Build multimodal dataset ---
ds = RactoDataset()

image_answer_pairs = [
    ("product_a.jpg", "A red ceramic mug with a black handle, 350ml capacity."),
    ("product_b.jpg", "A stainless steel water bottle, 750ml, matte finish."),
    ("product_c.jpg", "A glass pitcher with a bamboo lid, 1.2L capacity."),
    # ... hundreds more
]

for img_path, description in image_answer_pairs:
    ds.add(
        RactoTrainingExample.from_pair(
            user="Describe this product precisely.",
            assistant=description,
            system="You are a product cataloguer. Output one factual sentence.",
            user_attachments=[RactoFile.from_path(img_path)],
        )
    )

print(ds.summary())
# {"examples": 3, "total_messages": 9, "avg_turns_per_example": 3.0, "multimodal_examples": 3}

# --- 2. Validate and split ---
errors = ds.validate("openai")
assert not errors, errors

train_ds, val_ds = ds.split(0.85, seed=0)

# --- 3. Export for inspection ---
train_ds.export_jsonl("train.jsonl", provider="openai", overwrite=True)

# --- 4. Fine-tune (vision model) ---
tuner = OpenAIFineTuner()
fine_tuned_model = tuner.run_pipeline(
    train_ds,
    model="gpt-4o-2024-08-06",   # vision-capable base model
    validation_dataset=val_ds,
    n_epochs=3,
    suffix="product-cataloguer",
)

# --- 5. Use the fine-tuned model ---
prompt = RactoPrompt(
    role="You are a product cataloguer.",
    aim="Describe the product in the image.",
    constraints=["One sentence only.", "Include material, colour, and capacity."],
    tone="Factual",
    output_format="text",
)
kit = opd.OpenAIDeveloperKit(model=fine_tuned_model, default_prompt=prompt)
config = opd.ChatConfig(
    user_message="Describe this product.",
    attachments=[RactoFile.from_path("new_product.jpg")],
)
print(kit.chat(config).content)
```

---

### `RactoDataset` API Reference

| Member | Description |
| --- | --- |
| `RactoDataset.from_pairs(pairs, system="")` | Build from `(user, assistant)` text tuples |
| `RactoDataset.from_jsonl(path, provider)` | Load a previously exported JSONL file |
| `.add(example)` | Append one `RactoTrainingExample` |
| `.extend(examples)` | Append a list of examples |
| `.validate(provider)` | Returns `list[str]` of errors (empty = valid) |
| `.split(train_ratio, seed)` | Returns `(train_ds, val_ds)` |
| `.shuffle(seed)` | Returns a new shuffled dataset |
| `.export_jsonl(path, provider, overwrite)` | Write to `.jsonl` file on disk |
| `.to_jsonl_string(provider)` | Return JSONL as a `str` (no I/O) |
| `.summary()` | Dict with `examples`, `multimodal_examples`, etc. |

### `RactoTrainingExample` API Reference

| Member | Description |
| --- | --- |
| `RactoTrainingExample.from_pair(user, assistant, system, user_attachments)` | Single-turn factory |
| `RactoTrainingExample.from_conversation(turns)` | From `[(role, content), …]` list |
| `.to_openai_dict()` | `{"messages": […]}` for OpenAI JSONL |
| `.to_anthropic_dict()` | `{"system": "…", "messages": […]}` for Anthropic JSONL |
| `.to_gemini_dict()` | `{"text_input": …, "output": …}` or `{"contents": […]}` for Gemini |

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

## Architecture

```text
src/ractogateway/
├── __init__.py                          # Top-level: RactoPrompt, Gateway, tool, ToolRegistry
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
│   └── anthropic_tuner.py              #   AnthropicFineTuner
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
├── gateway/                             # Unified Gateway Runner
│   └── runner.py                        #   Gateway orchestrator class
│
├── openai_developer_kit/                # OpenAI Developer Kit (import as opd)
│   └── kit.py                           #   OpenAIDeveloperKit class
│
├── google_developer_kit/                # Google Developer Kit (import as god)
│   └── kit.py                           #   GoogleDeveloperKit class
│
└── anthropic_developer_kit/             # Anthropic Developer Kit (import as anth)
    └── kit.py                           #   AnthropicDeveloperKit class
```

### Design Principles

- **Lazy provider imports** — `openai`, `google-genai`, and `anthropic` SDKs are only imported when you instantiate a kit. `import ractogateway` never fails due to a missing optional dependency.
- **Composition over inheritance** — Developer kits compose internal adapters rather than extending them, keeping the public API surface clean.
- **Pydantic everywhere** — Every input is a validated model. Every output is a typed model. No `dict[str, Any]` at the API boundary.
- **Sync + async parity** — Every method has both a synchronous and asynchronous variant.
- **Provider-agnostic tool schemas** — Define tools once, use them with any provider. The internal adapters handle the translation.

---

## Environment Variables

| Variable | Provider | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI | API key (used when `api_key` not passed to constructor) |
| `GEMINI_API_KEY` | Google | API key (used when `api_key` not passed to constructor) |
| `ANTHROPIC_API_KEY` | Anthropic | API key (used when `api_key` not passed to constructor) |

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
