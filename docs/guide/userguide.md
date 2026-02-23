# RactoGateway — Complete User Guide

> **Who this guide is for:** complete beginners who have never used an LLM library before, as well as experienced developers who want a deep-dive reference. Every parameter is explained in plain English *and* in technical terms, with working code examples and expected output.

---

## Table of Contents

1. [Jargon Buster — Know the Words Before You Write the Code](#1-jargon-buster)
2. [What is RactoGateway?](#2-what-is-ractogateway)
3. [Installation](#3-installation)
4. [Core Mental Model](#4-core-mental-model)
5. [RactoPrompt — The Heart of Every Request](#5-ractoprompt)
6. [Developer Kits — Your Chat Interface](#6-developer-kits)
7. [Your First Chat](#7-your-first-chat)
8. [ChatConfig — Controlling Every Request](#8-chatconfig)
9. [Getting Structured / Typed Output](#9-structured-output)
10. [Multi-Turn Conversations (History)](#10-multi-turn-conversations)
11. [Streaming — Real-Time Token-by-Token Output](#11-streaming)
12. [Tool Calling — LLM Calls Your Python Functions](#12-tool-calling)
13. [File Attachments — Vision & PDFs](#13-file-attachments)
14. [Embeddings — Teaching Machines to Understand Text](#14-embeddings)
15. [Performance & Cost Optimisation](#15-performance--cost-optimisation)
    - 15.1 Exact Match Cache
    - 15.2 Semantic Cache
    - 15.3 Token Truncation
    - 15.4 Cost-Aware Routing
16. [Google & Anthropic Kits](#16-google--anthropic-kits)
17. [RAG — Retrieval-Augmented Generation](#17-rag--retrieval-augmented-generation)
18. [Redis — Production Infrastructure](#18-redis--production-infrastructure)
19. [Common Mistakes & How to Fix Them](#19-common-mistakes--how-to-fix-them)

---

## 1. Jargon Buster

Before diving into code, here are the key terms you will encounter. Skip to §2 if you already know these.

| Term | Plain-English Meaning | Technical Definition |
|---|---|---|
| **LLM** | A very powerful autocomplete that understands meaning | Large Language Model — a neural network trained on vast text corpora to predict/generate natural language |
| **Prompt** | What you say to the AI | The input text (plus optional instructions) sent to an LLM |
| **Completion / Response** | What the AI says back | The LLM's generated output tokens |
| **Token** | Roughly one word (sometimes less) | The smallest unit an LLM processes; ~4 chars for English |
| **System Prompt** | The AI's job description | An instruction block sent before the conversation; sets behaviour and constraints |
| **Temperature** | How creative vs. predictable the AI is | Float 0–2. 0 = deterministic (same output every time). Higher = more random/creative |
| **Streaming** | Getting the answer word-by-word in real time | Server-sent events where each token is pushed to the client as it is generated |
| **Embedding** | Converting text into a list of numbers | A dense vector representation where semantically similar texts are numerically close |
| **RAG** | Letting the AI "look things up" before answering | Retrieval-Augmented Generation — retrieve relevant chunks from a knowledge base and inject them into the prompt |
| **Tool Calling** | The AI can trigger your Python functions | Function-calling protocol where the LLM emits a structured intent and the client executes a real function |
| **Pydantic Model** | A Python class that validates data automatically | A `BaseModel` subclass that enforces types and field rules at runtime |
| **Cache** | Store an answer so you don't ask the AI twice | In-memory or distributed key-value store keyed on request fingerprint |
| **Context Window** | The AI's short-term memory | Maximum number of tokens the model can process in one request |
| **Adapter** | The translator between our library and the AI provider | A thin class that converts our internal format to the OpenAI / Google / Anthropic API wire format |

---

## 2. What is RactoGateway?

**Plain English:** RactoGateway is a Python library that lets you talk to different AI models (OpenAI, Google, Anthropic) using the same code. You don't need to learn three different APIs. You write your prompts using a structured template (the RACTO principle), and the library takes care of formatting, caching, routing, and more.

**Technical:** RactoGateway is a provider-agnostic LLM orchestration SDK built on Pydantic. It provides:

- A unified `RactoPrompt` structured prompt compiler (the RACTO principle)
- Provider-specific developer kits (`OpenAIDeveloperKit`, `GoogleDeveloperKit`, `AnthropicDeveloperKit`)
- Sync **and** async parity on every method
- Optional middleware: exact-match cache, semantic cache, cost-aware router, token truncator
- Tool calling, file attachments, streaming, embeddings, RAG, fine-tuning, and production infra (Redis, Celery, Kafka)

**Why does this exist?** Without RactoGateway, switching from OpenAI to Anthropic means rewriting all your code. With RactoGateway, you swap one class name.

---

## 3. Installation

```bash
# Minimum — no LLM provider yet
pip install ractogateway

# OpenAI (GPT models)
pip install "ractogateway[openai]"

# Google (Gemini models)
pip install "ractogateway[google]"

# Anthropic (Claude models)
pip install "ractogateway[anthropic]"

# All three providers at once
pip install "ractogateway[all]"

# RAG (document reading, chunking, embedding, stores)
pip install "ractogateway[rag-all]"

# Redis (distributed cache, rate limiting, chat memory)
pip install "ractogateway[redis]"
```

**Requires Python 3.10 or later.**

---

## 4. Core Mental Model

Think of RactoGateway in three layers:

```
┌─────────────────────────────────────────────────────┐
│  YOUR CODE                                          │
│  RactoPrompt → ChatConfig → kit.chat()              │
├─────────────────────────────────────────────────────┤
│  DEVELOPER KIT  (OpenAIDeveloperKit, etc.)           │
│  middleware: cache → route → truncate → API call    │
├─────────────────────────────────────────────────────┤
│  ADAPTER  (OpenAILLMKit, GoogleLLMKit, etc.)         │
│  Translates our format → provider wire format       │
├─────────────────────────────────────────────────────┤
│  PROVIDER API  (OpenAI, Google, Anthropic)           │
└─────────────────────────────────────────────────────┘
```

**You only ever touch the top layer.** The kit and adapter layers are managed for you.

---

## 5. RactoPrompt

`RactoPrompt` is how you write instructions for the AI. It enforces the **RACTO principle** — a structured format that dramatically reduces hallucinations and ambiguous outputs.

**RACTO stands for:**

| Letter | Field | Plain English | Technical |
|---|---|---|---|
| **R** | `role` | Who is the AI? | System identity; primes the model's behaviour via persona specification |
| **A** | `aim` | What should it do? | Objective statement; the task the model must complete |
| **C** | `constraints` | What must it never do? | Hard invariants; rule set injected into `[CONSTRAINTS]` block |
| **T** | `tone` | How should it talk? | Communication register; affects lexical and stylistic choices |
| **O** | `output_format` | What shape should the answer be in? | Output schema; can be a keyword, a string, or a Pydantic model class |

Plus two optional helpers: `context` (background knowledge) and `examples` (few-shot examples).

### 5.1 Minimal Example

```python
from ractogateway.prompts.engine import RactoPrompt

prompt = RactoPrompt(
    role="You are a helpful customer-support agent for a software company.",
    aim="Answer the user's question about our product.",
    constraints=[
        "Never make up features that don't exist.",
        "If you don't know the answer, say so.",
    ],
    tone="Friendly and concise.",
    output_format="text",
)

# See what the compiled system prompt looks like:
print(prompt.compile())
```

**Expected output:**

```
[ROLE]
You are a helpful customer-support agent for a software company.

[AIM]
Answer the user's question about our product.

[CONSTRAINTS]
- Never make up features that don't exist.
- If you don't know the answer, say so.

[TONE]
Friendly and concise.

[OUTPUT]
Respond in plain text with no special formatting.

[GUARDRAILS]
- If you are unsure or lack sufficient information, state it explicitly rather than guessing.
- Do NOT fabricate facts, citations, URLs, statistics, or code that you cannot verify.
- Stick strictly to what is asked. Do not add unrequested information.
- If the answer requires assumptions, list each assumption explicitly before proceeding.
```

> **Notice the `[GUARDRAILS]` section at the bottom.** This is auto-generated by `anti_hallucination=True` (the default). It tells the model to be honest about uncertainty. You can disable it with `anti_hallucination=False` if you need maximum creative freedom.

---

### 5.2 Full Parameter Reference

```python
from pydantic import BaseModel

class Summary(BaseModel):
    headline: str
    bullet_points: list[str]
    confidence_score: float  # 0.0 to 1.0

prompt = RactoPrompt(
    # ── REQUIRED ──────────────────────────────────────────────────────
    role="You are a senior financial analyst.",
    # Plain: "Tell the AI who it is"
    # Technical: Persona string prepended to the [ROLE] block; primes
    #            the model's prior distribution toward domain-specific vocabulary

    aim="Summarise the provided earnings report into key takeaways.",
    # Plain: "Tell the AI what job it has to do"
    # Technical: Task objective injected into [AIM]; should be one clear imperative sentence

    constraints=[
        "Only use numbers that appear in the report — never invent figures.",
        "Keep bullet points to at most 15 words each.",
        "Do not provide investment advice.",
    ],
    # Plain: "Red lines the AI must never cross"
    # Technical: List[str]; each item becomes a bullet in [CONSTRAINTS].
    #            Minimum one constraint required.

    tone="Professional, concise, and factual.",
    # Plain: "How the AI should sound"
    # Technical: Register specification injected into [TONE]; affects temperature
    #            interaction and lexical formality

    output_format=Summary,
    # Plain: "Exactly what shape should the answer be in?"
    # Technical: Union[str, type[BaseModel]].
    #   - "text"     → plain text
    #   - "json"     → raw JSON object
    #   - "markdown" → markdown-formatted response
    #   - A Pydantic model class → the full JSON Schema is embedded in the prompt;
    #     the LLM must return JSON that validates against it.

    # ── OPTIONAL ──────────────────────────────────────────────────────
    context="Q3 2025 earnings call. Revenue: $4.2B (+12% YoY). EPS: $1.87.",
    # Plain: "Background knowledge the AI needs to do its job"
    # Technical: Domain-specific text injected between [AIM] and [CONSTRAINTS].
    #            Ideal for passing documents, retrieved chunks, or facts.

    examples=[
        {
            "input":  "Revenue grew 5% but EPS fell 10%.",
            "output": '{"headline": "Mixed signals: top-line growth masked by margin compression", ...}'
        },
    ],
    # Plain: "Show the AI what a good answer looks like"
    # Technical: Few-shot exemplars injected into [EXAMPLES] block; each dict
    #            must contain exactly "input" and "output" keys.

    anti_hallucination=True,
    # Plain: "Should the AI be told to say 'I don't know' instead of guessing?"
    # Technical: Boolean flag. When True, appends [GUARDRAILS] block with
    #            explicit uncertainty-disclosure directives. Default: True.
)
```

---

## 6. Developer Kits

A **Developer Kit** is your interface to a specific LLM provider. All three kits (`OpenAIDeveloperKit`, `GoogleDeveloperKit`, `AnthropicDeveloperKit`) share the same method names.

### OpenAIDeveloperKit — Full Parameter Reference

```python
from ractogateway import openai_developer_kit as gpt

kit = gpt.OpenAIDeveloperKit(
    model="gpt-4o",
    # Plain: "Which AI model should I use?"
    # Technical: Chat model ID passed to openai.chat.completions.create(model=...).
    #            Use "auto" to enable cost-aware routing (requires router= param).
    #            Common values: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o3-mini"

    api_key="sk-...",
    # Plain: "My OpenAI account password"
    # Technical: Bearer token for OpenAI API auth. Falls back to
    #            os.environ["OPENAI_API_KEY"] when omitted.

    base_url=None,
    # Plain: "Send requests to a different server (e.g. Azure or your own proxy)"
    # Technical: Override for openai.base_url. Used for Azure OpenAI endpoints or
    #            local model servers that implement the OpenAI protocol.

    embedding_model="text-embedding-3-small",
    # Plain: "Which model to use when converting text to numbers (embeddings)"
    # Technical: Default model ID for embed() / aembed() calls.
    #            Passed to openai.embeddings.create(model=...).

    default_prompt=None,
    # Plain: "A prompt to use for every request unless I override it"
    # Technical: RactoPrompt instance used when ChatConfig.prompt is None.
    #            If both are None, kit.chat() raises ValueError.

    exact_cache=None,
    # Plain: "Store answers so I don't pay for the same question twice"
    # Technical: ExactMatchCache instance. On a byte-identical request the cached
    #            LLMResponse is returned without an API call. O(1) lookup.

    semantic_cache=None,
    # Plain: "Store answers and also reuse them for questions that mean the same thing"
    # Technical: SemanticCache instance. Uses cosine similarity on embeddings.
    #            Returns cached response when similarity >= threshold.

    router=None,
    # Plain: "Automatically pick the cheapest model that can handle each question"
    # Technical: CostAwareRouter instance. Routes each request to the first tier
    #            whose max_score >= the computed prompt complexity score.
    #            Required when model="auto".

    truncator=None,
    # Plain: "Automatically shorten old conversation history if it gets too long"
    # Technical: TokenTruncator instance. Trims history messages to keep total
    #            token count within the model's context window before each API call.
)
```

---

## 7. Your First Chat

Let's put it all together — a complete, working example.

```python
import os
from ractogateway import openai_developer_kit as gpt
from ractogateway.prompts.engine import RactoPrompt

# 1. Define who the AI is and what it should do
prompt = RactoPrompt(
    role="You are a helpful Python tutor.",
    aim="Explain the concept the user asks about in simple terms.",
    constraints=["Use beginner-friendly language.", "Keep the answer under 3 sentences."],
    tone="Warm, encouraging, and clear.",
    output_format="text",
)

# 2. Create the kit (reads OPENAI_API_KEY from environment automatically)
kit = gpt.OpenAIDeveloperKit(
    model="gpt-4o-mini",
    default_prompt=prompt,
)

# 3. Send a message and get a response
response = kit.chat(gpt.ChatConfig(user_message="What is a Python list?"))

print(response.content)
# A list in Python is an ordered collection of items that can hold any type
# of data — numbers, strings, even other lists. You create one with square
# brackets, like my_list = [1, "hello", True]. You can add, remove, or
# change items at any time!

print(f"Model used: {response.model}")
# Model used: gpt-4o-mini

print(f"Tokens used: {response.usage}")
# Tokens used: {'prompt_tokens': 127, 'completion_tokens': 54, 'total_tokens': 181}

print(f"Why did generation stop: {response.finish_reason}")
# Why did generation stop: FinishReason.STOP
```

### What is `LLMResponse`?

The return type of `kit.chat()` is an `LLMResponse` object. Here are its key fields:

| Field | Type | Plain English | Technical |
|---|---|---|---|
| `content` | `str \| None` | The AI's answer as a string | Raw text of the completion |
| `parsed` | `dict \| BaseModel \| None` | The answer as structured data (when using `response_model`) | JSON-decoded dict or validated Pydantic instance |
| `model` | `str` | Which model was actually used | Model ID echoed from the API response |
| `finish_reason` | `FinishReason` | Why the AI stopped generating | Enum: `STOP` (natural end), `LENGTH` (hit max_tokens), `TOOL_CALL` |
| `usage` | `dict[str, int]` | How many tokens were used | `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `tool_calls` | `list[ToolCallResult]` | Any tools the AI wanted to call | Non-empty when the model returns a function-call intent |
| `raw` | `Any` | The raw provider response object | Original SDK response; useful for debugging |

---

## 8. ChatConfig

`ChatConfig` is the object you pass to every `chat()`, `achat()`, `stream()`, and `astream()` call. It controls the details of a single request.

```python
from pydantic import BaseModel
from ractogateway import openai_developer_kit as gpt
from ractogateway.prompts.engine import RactoPrompt

class ProductReview(BaseModel):
    sentiment: str          # "positive" | "neutral" | "negative"
    score: int              # 1–10
    summary: str

config = gpt.ChatConfig(
    user_message="The keyboard is amazing but the battery dies in 3 hours.",
    # Plain: "The question or text you want to send to the AI"
    # Technical: The human turn content. Minimum 1 character (enforced by Pydantic).

    prompt=RactoPrompt(
        role="You are a product review classifier.",
        aim="Classify the review and return a structured analysis.",
        constraints=["Scores must be integers from 1 to 10."],
        tone="Neutral and objective.",
        output_format=ProductReview,
    ),
    # Plain: "Override the kit's default prompt for just this one request"
    # Technical: Per-request RactoPrompt. Takes precedence over kit.default_prompt.
    #            If both are None, raises ValueError.

    temperature=0.0,
    # Plain: "How predictable vs. creative should the answer be?"
    # Technical: Sampling temperature. Float in [0.0, 2.0].
    #   0.0 → argmax decoding (fully deterministic, same output for same input)
    #   ~0.7 → balanced creativity/coherence (good for most tasks)
    #   1.5+ → very random; may become incoherent for structured tasks

    max_tokens=512,
    # Plain: "Maximum length of the AI's answer"
    # Technical: Hard cap on completion tokens. If the model hasn't finished,
    #            generation stops and finish_reason becomes LENGTH.
    #            Default is 4096. Keep lower for short structured tasks to save cost.

    response_model=ProductReview,
    # Plain: "Validate the AI's JSON answer against this Python class"
    # Technical: type[BaseModel]. After the API call, the raw JSON content is
    #            parsed and validated via ProductReview.model_validate().
    #            On failure, a warning is appended to response.content — no exception.

    history=[],
    # Plain: "Previous messages in the conversation (for multi-turn chat)"
    # Technical: list[Message]. Each Message has role (user/assistant/system) and
    #            content (str). Injected between the system prompt and the current
    #            user message. Managed manually or via RedisChatMemory.

    tools=None,
    # Plain: "Python functions the AI is allowed to call"
    # Technical: ToolRegistry instance. The adapter serialises its schemas into
    #            provider-specific function-calling format before the API call.

    extra={},
    # Plain: "Any other provider-specific settings I want to pass"
    # Technical: Pass-through dict merged into the API request kwargs.
    #            E.g. extra={"seed": 42, "top_p": 0.9, "stop": ["\n\n"]}
)

response = kit.chat(config)
print(response.parsed)
# {'sentiment': 'neutral', 'score': 5, 'summary': 'Great keyboard but very poor battery life.'}
```

---

## 9. Structured Output

One of the most powerful features: getting a validated Python object back from the AI instead of raw text.

### Step 1 — Define your output shape with Pydantic

```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    condition: str          # e.g. "sunny", "rainy", "cloudy"
    uv_index: int
```

### Step 2 — Pass the class as `output_format` in RactoPrompt

```python
from ractogateway.prompts.engine import RactoPrompt

prompt = RactoPrompt(
    role="You are a weather data formatter.",
    aim="Parse the user's description into a structured weather report.",
    constraints=["Always use Celsius.", "UV index must be 0–11."],
    tone="Concise and data-focused.",
    output_format=WeatherReport,   # <-- the Pydantic class
)
```

### Step 3 — Also pass it as `response_model` in ChatConfig

```python
from ractogateway import openai_developer_kit as gpt

kit = gpt.OpenAIDeveloperKit(model="gpt-4o-mini", default_prompt=prompt)

config = gpt.ChatConfig(
    user_message="London, 18 degrees, overcast, UV 3.",
    response_model=WeatherReport,   # <-- validates the parsed JSON
)

response = kit.chat(config)

# response.parsed is a dict already validated against WeatherReport
print(response.parsed)
# {'city': 'London', 'temperature_celsius': 18.0, 'condition': 'overcast', 'uv_index': 3}

# To get a proper WeatherReport instance:
report = WeatherReport(**response.parsed)
print(report.city)           # London
print(report.uv_index)       # 3
print(type(report))          # <class '__main__.WeatherReport'>
```

> **Why two places?** `output_format` in `RactoPrompt` tells the LLM what to generate (embeds the JSON Schema in the system prompt). `response_model` in `ChatConfig` validates the output in Python. Use both together for end-to-end type safety.

---

## 10. Multi-Turn Conversations

To have a conversation with memory, pass the `history` list to each `ChatConfig`:

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway._models.chat import Message, MessageRole
from ractogateway.prompts.engine import RactoPrompt

kit = gpt.OpenAIDeveloperKit(
    model="gpt-4o-mini",
    default_prompt=RactoPrompt(
        role="You are a helpful AI assistant.",
        aim="Carry on a friendly conversation.",
        constraints=["Remember what the user said earlier."],
        tone="Casual and friendly.",
        output_format="text",
    ),
)

# Turn 1
response1 = kit.chat(gpt.ChatConfig(user_message="My name is Alice."))
print(response1.content)
# Nice to meet you, Alice! How can I help you today?

# Build the history from turn 1
history = [
    Message(role=MessageRole.USER, content="My name is Alice."),
    Message(role=MessageRole.ASSISTANT, content=response1.content),
]

# Turn 2 — the model now "remembers" turn 1
response2 = kit.chat(gpt.ChatConfig(
    user_message="What is my name?",
    history=history,   # <-- inject previous turns
))
print(response2.content)
# Your name is Alice! 😊
```

**Tip:** For production multi-user apps, use `RedisChatMemory` (see §18) to store history in Redis so it survives server restarts.

---

## 11. Streaming

Streaming lets you display the AI's answer word-by-word as it is generated — much better UX than waiting for the full response.

### Synchronous Streaming

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.prompts.engine import RactoPrompt

kit = gpt.OpenAIDeveloperKit(
    model="gpt-4o-mini",
    default_prompt=RactoPrompt(
        role="You are a storyteller.",
        aim="Write a short story based on the user's prompt.",
        constraints=["Keep it under 100 words."],
        tone="Vivid and imaginative.",
        output_format="text",
    ),
)

config = gpt.ChatConfig(user_message="A robot discovers it can dream.")

for chunk in kit.stream(config):
    # chunk.delta.text is the new text in this chunk (may be empty string)
    print(chunk.delta.text, end="", flush=True)

    if chunk.is_final:
        print()  # newline after the story
        print(f"Finish reason: {chunk.finish_reason}")
        print(f"Total tokens:  {chunk.usage.get('total_tokens', '?')}")
```

**Expected output (streaming, printed token-by-token):**

```
In the hum of the server room, Unit-7 closed its optical sensors...
and dreamed of open fields and laughter it had never known.
When it woke, it understood why humans called sleep a gift.

Finish reason: FinishReason.STOP
Total tokens:  112
```

### Asynchronous Streaming

```python
import asyncio
from ractogateway import openai_developer_kit as gpt

async def main():
    async for chunk in kit.astream(config):
        print(chunk.delta.text, end="", flush=True)
        if chunk.is_final:
            break

asyncio.run(main())
```

### What is `StreamChunk`?

| Field | Plain English | Technical |
|---|---|---|
| `delta.text` | New text arrived in this chunk | Incremental token string from the current event |
| `accumulated_text` | Everything generated so far | Concatenation of all previous `delta.text` values |
| `is_final` | Is this the last chunk? | `True` when `finish_reason` is set |
| `finish_reason` | Why did generation end? | `FinishReason.STOP`, `LENGTH`, or `TOOL_CALL` |
| `usage` | Token counts (only in final chunk) | Dict with `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `tool_calls` | Tools the model wants to call | Non-empty list when `finish_reason == TOOL_CALL` |
| `parsed` | Parsed + validated object (if `response_model` set) | Available on final chunk only |

---

## 12. Tool Calling

Tool calling lets the LLM trigger your Python functions. Useful for giving AI access to live data (weather, databases, calculators, etc.).

### Step 1 — Define your tools with the `@tool` decorator

```python
from ractogateway.tools.registry import tool, ToolRegistry

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    city: The name of the city to look up.
    unit: Temperature unit — 'celsius' or 'fahrenheit'.
    """
    # In a real app, call a weather API here
    return f"The weather in {city} is 22°{'C' if unit == 'celsius' else 'F'} and sunny."

@tool
def get_time(timezone: str) -> str:
    """Return the current time in the given timezone.

    timezone: IANA timezone string, e.g. 'Europe/London'.
    """
    from datetime import datetime
    import zoneinfo
    tz = zoneinfo.ZoneInfo(timezone)
    return datetime.now(tz).strftime("%H:%M on %A, %d %B %Y")
```

### Step 2 — Register them in a `ToolRegistry`

```python
registry = ToolRegistry()
registry.register(get_weather)
registry.register(get_time)
```

### Step 3 — Pass the registry to `ChatConfig`

```python
from ractogateway.prompts.engine import RactoPrompt
from ractogateway import openai_developer_kit as gpt

kit = gpt.OpenAIDeveloperKit(
    model="gpt-4o",
    default_prompt=RactoPrompt(
        role="You are a helpful assistant with access to live data tools.",
        aim="Answer the user's question using the available tools.",
        constraints=["Always use the tools when relevant."],
        tone="Helpful and precise.",
        output_format="text",
    ),
)

config = gpt.ChatConfig(
    user_message="What's the weather like in Paris and what time is it there?",
    tools=registry,
)

response = kit.chat(config)

# If the model decided to call tools:
if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Tool requested: {tc.name}")
        print(f"Arguments:      {tc.arguments}")

        # Execute the actual Python function
        fn = registry.get_callable(tc.name)
        if fn:
            result = fn(**tc.arguments)
            print(f"Result:         {result}")
```

**Expected output:**

```
Tool requested: get_weather
Arguments:      {'city': 'Paris', 'unit': 'celsius'}
Result:         The weather in Paris is 22°C and sunny.

Tool requested: get_time
Arguments:      {'timezone': 'Europe/Paris'}
Result:         14:32 on Monday, 24 February 2026
```

> **What is `ToolCallResult`?** It has three fields: `id` (unique call ID from the API), `name` (function name), and `arguments` (dict of parsed arguments ready to `**unpack` into your function).

---

## 13. File Attachments

Send images, PDFs, and text files alongside your text message using `RactoFile`.

```python
from ractogateway.prompts.engine import RactoPrompt, RactoFile
from ractogateway import openai_developer_kit as gpt

kit = gpt.OpenAIDeveloperKit(
    model="gpt-4o",   # must be a vision-capable model
    default_prompt=RactoPrompt(
        role="You are a visual QA assistant.",
        aim="Describe what you see in the attached image.",
        constraints=["Be specific about colours, shapes, and text visible in the image."],
        tone="Descriptive and precise.",
        output_format="text",
    ),
)

# Load an image from disk (MIME type is auto-detected)
image = RactoFile.from_path("/path/to/screenshot.png")

# Or from raw bytes:
# image = RactoFile.from_bytes(open("photo.jpg","rb").read(), "image/jpeg")

messages = prompt.to_messages(
    user_message="What is shown in this image?",
    attachments=[image],
    provider="openai",   # formats content blocks for the correct provider
)

# You can also just use kit.chat() with a ChatConfig — attachments can be
# baked into the prompt's to_messages() call directly
```

### `RactoFile` Parameter Reference

| Method / Param | Plain English | Technical |
|---|---|---|
| `RactoFile.from_path(path)` | Load a file from your disk | Reads bytes and auto-detects MIME type via `mimetypes.guess_type` |
| `RactoFile.from_bytes(data, mime_type)` | Create from raw bytes you already have | No disk I/O; pass `bytes` + an explicit MIME type string |
| `data` | The file's raw bytes | `bytes` object |
| `mime_type` | What type of file it is | MIME string: `"image/png"`, `"image/jpeg"`, `"application/pdf"`, `"text/plain"`, etc. |
| `name` | An optional filename label | `str`; used for display/debugging only |
| `is_image` | Is it a picture? | `True` for JPEG, PNG, GIF, WEBP |
| `is_pdf` | Is it a PDF? | `True` for `application/pdf` |
| `base64_data` | File as a base64 string | Used internally by the provider adapters |

---

## 14. Embeddings

Embeddings convert text into lists of numbers (vectors) where semantically similar texts end up numerically close. This powers semantic search, clustering, and RAG.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway._models.embedding import EmbeddingConfig

kit = gpt.OpenAIDeveloperKit(model="gpt-4o-mini")

config = EmbeddingConfig(
    texts=["Python is a programming language.", "I love apples.", "Java is also a language."],
    # Plain: "The list of strings to convert into number vectors"
    # Technical: List[str] passed to openai.embeddings.create(input=...)

    model="text-embedding-3-small",
    # Plain: "Which embedding model to use"
    # Technical: Overrides kit.embedding_model for this specific call.
    #            None means use the kit's default.

    dimensions=None,
    # Plain: "How many numbers should each vector have?"
    # Technical: Optional int. For text-embedding-3-*, you can reduce from 1536
    #            to a smaller size (e.g. 256) for faster similarity search.
)

response = kit.embed(config)

for vec in response.vectors:
    print(f"Text:    {vec.text!r}")
    print(f"Index:   {vec.index}")
    print(f"Vector:  [{vec.embedding[0]:.4f}, {vec.embedding[1]:.4f}, ...]  (length {len(vec.embedding)})")
    print()
```

**Expected output:**

```
Text:    'Python is a programming language.'
Index:   0
Vector:  [0.0123, -0.0456, ...]  (length 1536)

Text:    'I love apples.'
Index:   1
Vector:  [-0.0234, 0.0789, ...]  (length 1536)

Text:    'Java is also a language.'
Index:   2
Vector:  [0.0118, -0.0451, ...]  (length 1536)
```

> **Pro tip:** Texts 0 and 2 will have very similar vectors because they are semantically related ("programming languages"). Text 1 will be far from both. This is the essence of embedding-powered semantic search.

---

## 15. Performance & Cost Optimisation

### 15.1 Exact Match Cache

**Plain English:** If someone asks the exact same question again (same words, same settings), return the cached answer instantly — no API call, no cost.

**Technical:** SHA-256 keyed over `(user_message, system_prompt, model, temperature, max_tokens)`. LRU eviction with optional TTL. Thread-safe via `threading.Lock`.

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.cache import ExactMatchCache

cache = ExactMatchCache(
    max_size=1024,
    # Plain: "How many answers to remember at most"
    # Technical: LRU capacity. When full, the least-recently-used entry is evicted.
    #            0 = unlimited (no eviction ever).

    ttl_seconds=3600,
    # Plain: "Forget an answer after this many seconds"
    # Technical: Float. Entries older than ttl_seconds are treated as cache misses
    #            and lazily evicted on next access. None = never expire.
)

kit = gpt.OpenAIDeveloperKit(model="gpt-4o-mini", exact_cache=cache)

# First call — hits the API
r1 = kit.chat(gpt.ChatConfig(user_message="What is the capital of France?"))
print(r1.content)   # Paris is the capital of France.

# Second call (identical) — served from cache in microseconds, $0 cost
r2 = kit.chat(gpt.ChatConfig(user_message="What is the capital of France?"))
print(r2.content)   # Paris is the capital of France.

print(cache.stats)  # CacheStats(hits=1, misses=1, size=1)
```

---

### 15.2 Semantic Cache

**Plain English:** Even if the question is *worded differently*, return the cached answer if it means the same thing.

**Technical:** Embeds each new query and computes cosine similarity against stored embeddings. Returns the cached response when similarity ≥ threshold.

```python
from ractogateway.cache import SemanticCache
import ractogateway.openai_developer_kit as gpt

# You supply an embedding function — any callable (str) -> list[float]
kit_for_embed = gpt.OpenAIDeveloperKit(model="gpt-4o-mini")

def embed(text: str) -> list[float]:
    from ractogateway._models.embedding import EmbeddingConfig
    resp = kit_for_embed.embed(EmbeddingConfig(texts=[text]))
    return resp.vectors[0].embedding

sem_cache = SemanticCache(
    embedder=embed,
    # Plain: "A function that converts text to a list of numbers"
    # Technical: Callable[[str], list[float]]. Called once for each new query
    #            to compute its embedding for similarity comparison.

    similarity_threshold=0.92,
    # Plain: "How similar does a question have to be to reuse a cached answer?"
    # Technical: Float in (0, 1]. Cosine similarity minimum. Higher = stricter match.
    #            0.92 works well; lower (e.g. 0.85) gives more cache hits but may
    #            return wrong answers for loosely-related questions.

    max_size=512,
    # Plain: "How many answers to remember"
    # Technical: LRU capacity for the semantic cache store.
)

kit = gpt.OpenAIDeveloperKit(model="gpt-4o-mini", semantic_cache=sem_cache)

# First call
r1 = kit.chat(gpt.ChatConfig(user_message="What is the capital of France?"))
# → API call happens

# Different wording, same meaning — cache HIT (if similarity >= 0.92)
r2 = kit.chat(gpt.ChatConfig(user_message="Which city is France's capital?"))
# → No API call; cached answer returned
```

---

### 15.3 Token Truncation

**Plain English:** Long conversations can overflow the AI's memory limit. The truncator automatically cuts old messages to keep things within bounds.

**Technical:** Sliding-window strategy over `ChatConfig.history`. Keeps `keep_first_n` messages and `keep_last_n` messages; drops the middle. Uses `len(text) // 4` as a token estimator by default, or `tiktoken` for precision.

```python
from ractogateway.truncation import TokenTruncator, TruncationConfig, MODEL_CONTEXT_LIMITS
from ractogateway import openai_developer_kit as gpt

truncator = TokenTruncator(TruncationConfig(
    keep_first_n=2,
    # Plain: "Always keep the first N history messages (e.g. important instructions)"
    # Technical: int. These messages are never evicted, regardless of token count.

    keep_last_n=8,
    # Plain: "Always keep the most recent N messages"
    # Technical: int. Recent context is preserved; only the 'middle' is dropped.

    safety_margin=512,
    # Plain: "Leave room for the model's reply"
    # Technical: Tokens reserved for the completion. Effective limit =
    #            context_window - safety_margin.

    token_counter=None,
    # Plain: "How to count tokens (leave blank for fast estimate)"
    # Technical: Optional Callable[[str], int]. When None, uses len(text) // 4.
    #            For precision, pass tiktoken: lambda t: len(enc.encode(t))
))

kit = gpt.OpenAIDeveloperKit(model="gpt-4o", truncator=truncator)
# Now every kit.chat() / kit.achat() call will auto-trim history before sending.

# Check the context limit for any model:
print(MODEL_CONTEXT_LIMITS["gpt-4o"])         # 128000
print(MODEL_CONTEXT_LIMITS["gpt-4o-mini"])    # 128000
print(MODEL_CONTEXT_LIMITS["claude-opus-4-6"])  # 200000
```

---

### 15.4 Cost-Aware Routing

**Plain English:** Not every question needs the most expensive model. Automatically send simple questions to a cheap model and hard questions to a powerful one.

**Technical:** Scores each prompt (0–100) based on length, question complexity markers, and keyword signals. Routes to the first `RoutingTier` whose `max_score >= score`. Adapters are pooled for O(1) model switching.

```python
from ractogateway.routing import CostAwareRouter, RoutingTier
from ractogateway import openai_developer_kit as gpt

router = CostAwareRouter([
    RoutingTier(
        model="gpt-4o-mini",
        max_score=30,
        # Plain: "Use this cheap model for easy questions (score 0–30)"
        # Technical: First tier. model= is the ID passed to the adapter.
        #            max_score= is the upper bound of the score range this tier handles.
    ),
    RoutingTier(
        model="gpt-4o",
        max_score=70,
        # Plain: "Use this mid-tier model for moderate questions (score 31–70)"
    ),
    RoutingTier(
        model="o3-mini",
        max_score=100,
        # Plain: "Use this powerful (expensive) model for hard questions (score 71–100)"
        # Technical: Final tier; also the fallback if no earlier tier matches.
    ),
])

kit = gpt.OpenAIDeveloperKit(
    model="auto",    # <-- REQUIRED when using a router
    router=router,
)

# "2+2" → very low complexity score → routed to gpt-4o-mini (cheapest)
r1 = kit.chat(gpt.ChatConfig(user_message="What is 2 + 2?"))
print(r1.model)   # gpt-4o-mini

# Complex reasoning → high score → routed to o3-mini
r2 = kit.chat(gpt.ChatConfig(
    user_message=(
        "Explain the mathematical proof of Gödel's incompleteness theorem "
        "and its implications for formal systems and computability theory."
    )
))
print(r2.model)   # o3-mini
```

### Combining All Middleware

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.cache import ExactMatchCache, SemanticCache
from ractogateway.routing import CostAwareRouter, RoutingTier
from ractogateway.truncation import TokenTruncator, TruncationConfig

kit = gpt.OpenAIDeveloperKit(
    model="auto",
    router=CostAwareRouter([
        RoutingTier(model="gpt-4o-mini", max_score=30),
        RoutingTier(model="gpt-4o",      max_score=100),
    ]),
    exact_cache=ExactMatchCache(max_size=2048, ttl_seconds=7200),
    semantic_cache=SemanticCache(embedder=embed, similarity_threshold=0.90),
    truncator=TokenTruncator(TruncationConfig(keep_last_n=10, safety_margin=1024)),
)
# Each request flows: exact cache → semantic cache → route → truncate → API call
```

---

## 16. Google & Anthropic Kits

The exact same patterns work with Google and Anthropic — just import the right kit.

### GoogleDeveloperKit (Gemini)

```python
from ractogateway import google_developer_kit as gemini
from ractogateway.prompts.engine import RactoPrompt

kit = gemini.GoogleDeveloperKit(
    model="gemini-2.0-flash",    # or "gemini-2.0-pro"
    api_key="AIza...",           # or set GOOGLE_API_KEY env var
)

prompt = RactoPrompt(
    role="You are a creative writing assistant.",
    aim="Write a haiku about the given subject.",
    constraints=["Must follow 5-7-5 syllable structure."],
    tone="Poetic and thoughtful.",
    output_format="text",
)

response = kit.chat(gemini.ChatConfig(
    user_message="Write a haiku about rain.",
    prompt=prompt,
))
print(response.content)
# Silver drops descend —
# Earth drinks its ancient thirst deep.
# Mud sings after rain.
```

### AnthropicDeveloperKit (Claude)

```python
from ractogateway import anthropic_developer_kit as claude
from ractogateway.prompts.engine import RactoPrompt

kit = claude.AnthropicDeveloperKit(
    model="claude-sonnet-4-6",   # or "claude-opus-4-6", "claude-haiku-4-5-20251001"
    api_key="sk-ant-...",        # or set ANTHROPIC_API_KEY env var
)

prompt = RactoPrompt(
    role="You are an expert code reviewer.",
    aim="Review the code snippet and identify any bugs or improvements.",
    constraints=["Be specific — cite line numbers.", "Prioritise correctness over style."],
    tone="Technical and direct.",
    output_format="markdown",
)

response = kit.chat(claude.ChatConfig(
    user_message="def divide(a, b): return a / b",
    prompt=prompt,
))
print(response.content)
```

**All three kits share the same methods:** `chat()`, `achat()`, `stream()`, `astream()`, `embed()`, `aembed()`.

---

## 17. RAG — Retrieval-Augmented Generation

**Plain English:** RAG lets the AI answer questions about your own documents. You feed it your files, it converts them into searchable number vectors, and when someone asks a question, it finds the relevant parts and feeds them to the AI.

**Technical:** Full pipeline: `FileReaderRegistry` → chunker → `ProcessingPipeline` → embedder → vector store → similarity search → `RactoPrompt` context injection.

### Complete RAG Pipeline Example

```python
from ractogateway.rag import RactoRAG
from ractogateway.rag.embedders import OpenAIEmbedder
from ractogateway.rag.stores import InMemoryVectorStore
from ractogateway.rag.chunkers import RecursiveChunker
from ractogateway import openai_developer_kit as gpt
from ractogateway.prompts.engine import RactoPrompt

# 1. Build the RAG pipeline
rag = RactoRAG(
    embedder=OpenAIEmbedder(api_key="sk-..."),
    store=InMemoryVectorStore(),   # swap for ChromaStore, FAISSStore, etc. in production
    chunker=RecursiveChunker(chunk_size=512, overlap=64),
)

# 2. Ingest your documents
rag.add_documents([
    "/path/to/product_manual.pdf",
    "/path/to/faq.docx",
    "/path/to/release_notes.txt",
])

# 3. At query time, retrieve relevant chunks
results = rag.retrieve("How do I reset my password?", top_k=3)

# 4. Inject retrieved context into a RactoPrompt
context = "\n\n".join(r.chunk.text for r in results)

prompt = RactoPrompt(
    role="You are a product support assistant.",
    aim="Answer the user's question based strictly on the provided documentation.",
    constraints=["Only use information from the CONTEXT section.", "Quote the source if possible."],
    tone="Helpful and precise.",
    output_format="text",
    context=context,    # <-- the retrieved chunks go here
)

# 5. Ask the AI
kit = gpt.OpenAIDeveloperKit(model="gpt-4o-mini", default_prompt=prompt)
response = kit.chat(gpt.ChatConfig(user_message="How do I reset my password?"))
print(response.content)
```

### Chunkers Explained

| Chunker | Plain English | Best For |
|---|---|---|
| `FixedChunker` | Split every N characters, no mercy | Quick prototyping, structured data |
| `RecursiveChunker` | Split at sentence/paragraph boundaries, then fall back to characters | General documents (best default) |
| `SentenceChunker` | Always split at sentence boundaries | Articles, legal text, Q&A content |
| `SemanticChunker` | Group sentences that are about the same topic | Complex documents with topic shifts |

### Vector Stores Explained

| Store | Plain English | When to Use |
|---|---|---|
| `InMemoryVectorStore` | Fast in-RAM store; lost on restart | Development, prototyping, tests |
| `ChromaStore` | Local persistent store | Single-server apps, local dev |
| `FAISSStore` | Facebook's ultra-fast similarity search | Millions of vectors, CPU-only |
| `PineconeStore` | Fully managed cloud vector DB | Production, no infra to manage |
| `QdrantStore` | Open-source, filterable, scalable | Production with metadata filtering |
| `WeaviateStore` | Open-source with built-in ML | Multi-modal + graph features |
| `MilvusStore` | Distributed vector DB | Billions of vectors at scale |
| `PGVectorStore` | PostgreSQL extension | Already using Postgres |

---

## 18. Redis — Production Infrastructure

Redis tools make your app production-ready: distributed cache, per-user rate limiting, and persistent chat memory that survives deployments.

```bash
pip install "ractogateway[redis]"
```

### 18.1 Distributed Exact Cache

Drop-in replacement for `ExactMatchCache` that works across multiple server replicas.

```python
from ractogateway.redis import RedisExactCache
from ractogateway import openai_developer_kit as gpt

cache = RedisExactCache(
    url="redis://localhost:6379/0",
    # Plain: "Where is your Redis server?"
    # Technical: Redis connection URL. Alternatively pass client= with a pre-built
    #            redis.Redis instance.

    ttl_seconds=3600,
    # Plain: "Forget cached answers after 1 hour"
    # Technical: TTL applied via Redis EXPIRE on each key write.
)

kit = gpt.OpenAIDeveloperKit(model="gpt-4o", exact_cache=cache)
# Now all your servers share the same cache!
```

### 18.2 Rate Limiter

Prevent users from making too many expensive requests.

```python
from ractogateway.redis import RedisRateLimiter, RateLimitConfig

limiter = RedisRateLimiter(
    url="redis://localhost:6379/0",
    config=RateLimitConfig(
        max_tokens_per_minute=5_000,
        # Plain: "Each user can use at most 5,000 tokens per minute"
        # Technical: Sliding 1-minute window. Counter stored as Redis sorted set per user_id.

        key_prefix="rl:",
        # Plain: "A label to group all rate limit keys in Redis"
        # Technical: String prefix for Redis keys: "{key_prefix}{user_id}"
    ),
)

# In your request handler:
user_id = "user-42"
estimated_tokens = 200

if not limiter.check_and_consume(user_id, tokens=estimated_tokens):
    raise RuntimeError("Rate limit exceeded — please try again in a minute.")

remaining = limiter.get_remaining(user_id)
print(f"Tokens remaining this minute: {remaining}")
# Tokens remaining this minute: 4800
```

### 18.3 Chat Memory

Store conversation history in Redis so it survives server restarts and scales across replicas.

```python
from ractogateway.redis import RedisChatMemory, ChatMemoryConfig
from ractogateway._models.chat import Message, MessageRole

memory = RedisChatMemory(
    url="redis://localhost:6379/0",
    config=ChatMemoryConfig(
        max_turns=20,
        # Plain: "Remember the last 20 messages per conversation"
        # Technical: Redis List capped to 2*max_turns entries (each turn = 2 messages).
        #            Older messages are popped from the front automatically.

        ttl_seconds=1800,
        # Plain: "Forget the conversation after 30 minutes of inactivity"
        # Technical: TTL reset on every append() call.

        key_prefix="chat:",
        # Plain: "Label all conversation keys in Redis"
        # Technical: Redis keys = "{key_prefix}{conv_id}"
    ),
)

# When a user sends a message:
conv_id = "session-abc123"
memory.append(conv_id, "user", "What's the best way to learn Python?")

# After getting the AI response:
memory.append(conv_id, "assistant", "Start with the official tutorial, then build projects!")

# Reconstruct history for the next request:
history_dicts = memory.get_history(conv_id)
# [{"role": "user", "content": "What's the best way..."}, {"role": "assistant", "content": "..."}]

history = [Message(role=m["role"], content=m["content"]) for m in history_dicts]

# Pass to ChatConfig:
response = kit.chat(gpt.ChatConfig(
    user_message="What resources do you recommend?",
    history=history,
))

# Wipe the conversation when the session ends:
memory.clear(conv_id)
print(memory.count(conv_id))  # 0
```

---

## 19. Common Mistakes & How to Fix Them

### Mistake 1: Using `output` instead of `output_format` in RactoPrompt

```python
# WRONG — this will raise a Pydantic ValidationError
prompt = RactoPrompt(
    role="...", aim="...", constraints=["..."], tone="...",
    output="text",    # ❌  field is called output_format, not output!
)

# CORRECT
prompt = RactoPrompt(
    role="...", aim="...", constraints=["..."], tone="...",
    output_format="text",   # ✅
)
```

### Mistake 2: Forgetting at least one constraint

```python
# WRONG — constraints cannot be an empty list
prompt = RactoPrompt(
    role="...", aim="...", constraints=[],   # ❌ ValidationError: min_length=1
    tone="...", output_format="text",
)

# CORRECT
prompt = RactoPrompt(
    role="...", aim="...",
    constraints=["Be helpful."],   # ✅ at least one constraint required
    tone="...", output_format="text",
)
```

### Mistake 3: Using `model="auto"` without a router

```python
# WRONG — raises ValueError immediately
kit = gpt.OpenAIDeveloperKit(model="auto")   # ❌

# CORRECT
kit = gpt.OpenAIDeveloperKit(
    model="auto",
    router=CostAwareRouter([...]),   # ✅
)
```

### Mistake 4: Neither ChatConfig.prompt nor kit.default_prompt is set

```python
# WRONG — raises ValueError when chat() is called
kit = gpt.OpenAIDeveloperKit(model="gpt-4o-mini")   # no default_prompt
response = kit.chat(gpt.ChatConfig(user_message="Hello"))  # ❌

# FIX OPTION 1: Set default_prompt on the kit
kit = gpt.OpenAIDeveloperKit(model="gpt-4o-mini", default_prompt=my_prompt)

# FIX OPTION 2: Pass prompt in ChatConfig
response = kit.chat(gpt.ChatConfig(user_message="Hello", prompt=my_prompt))
```

### Mistake 5: Not passing `response_model` to `ChatConfig` when you expect a typed object

```python
# WRONG — response.parsed will be a dict, NOT a validated WeatherReport instance
prompt = RactoPrompt(..., output_format=WeatherReport)
config = gpt.ChatConfig(user_message="...")   # ❌ missing response_model=

# CORRECT — also validate in ChatConfig
config = gpt.ChatConfig(
    user_message="...",
    response_model=WeatherReport,   # ✅
)
```

### Mistake 6: Missing `await` on async methods

```python
# WRONG — this returns a coroutine object, not a response
response = kit.achat(config)   # ❌

# CORRECT
response = await kit.achat(config)   # ✅  (inside an async function)
```

### Mistake 7: Not installing the provider extra

```python
# WRONG — if you only ran  pip install ractogateway
from ractogateway import openai_developer_kit as gpt
kit = gpt.OpenAIDeveloperKit(model="gpt-4o")
kit.chat(...)   # ❌  ImportError: The 'openai' package is required

# FIX
# pip install "ractogateway[openai]"
```

---

## Quick Reference Card

```python
# ── Imports ──────────────────────────────────────────────────────────
from ractogateway import openai_developer_kit as gpt
from ractogateway.prompts.engine import RactoPrompt, RactoFile
from ractogateway.tools.registry import tool, ToolRegistry
from ractogateway.cache import ExactMatchCache, SemanticCache
from ractogateway.routing import CostAwareRouter, RoutingTier
from ractogateway.truncation import TokenTruncator, TruncationConfig

# ── Build a prompt ───────────────────────────────────────────────────
prompt = RactoPrompt(
    role="...", aim="...", constraints=["..."], tone="...",
    output_format="text",    # or "json", "markdown", or a Pydantic class
    context="...",           # optional background knowledge
    examples=[{"input": "...", "output": "..."}],  # optional few-shot
)

# ── Create the kit ───────────────────────────────────────────────────
kit = gpt.OpenAIDeveloperKit(
    model="gpt-4o-mini",
    default_prompt=prompt,
    exact_cache=ExactMatchCache(max_size=512),
)

# ── Sync chat ────────────────────────────────────────────────────────
response = kit.chat(gpt.ChatConfig(user_message="Hello!"))
print(response.content)

# ── Async chat ───────────────────────────────────────────────────────
response = await kit.achat(gpt.ChatConfig(user_message="Hello!"))

# ── Streaming ────────────────────────────────────────────────────────
for chunk in kit.stream(gpt.ChatConfig(user_message="Tell me a story.")):
    print(chunk.delta.text, end="", flush=True)

# ── Embeddings ───────────────────────────────────────────────────────
from ractogateway._models.embedding import EmbeddingConfig
resp = kit.embed(EmbeddingConfig(texts=["hello", "world"]))
vec = resp.vectors[0].embedding   # list[float]

# ── Tool calling ─────────────────────────────────────────────────────
@tool
def get_price(product: str) -> float:
    """Get the price of a product."""
    return 9.99

registry = ToolRegistry()
registry.register(get_price)
response = kit.chat(gpt.ChatConfig(user_message="How much is a widget?", tools=registry))
```
