# RactoGateway

**One Python package for production-grade LLM workflows.**

RactoGateway gives you one clean SDK for:
- OpenAI
- Google Gemini
- Anthropic Claude
- Tool calling
- Structured outputs
- Streaming
- Embeddings
- RAG
- Fine-tuning datasets and tuners

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/ractogateway/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-GitHub-green.svg)](https://github.com/IAMPathak2702/RactoGateway)

---

## Why RactoGateway? (Teaching Note)

### Input (without RactoGateway)
```python
# Pseudo-typical code without a unified SDK:
# - Provider-specific request shapes
# - Provider-specific response parsing
# - Manual JSON-fence cleanup
# - Manual tool schema translation

raw_text = """```json
{"answer":"42"}
```"""
clean = raw_text.replace("```json", "").replace("```", "").strip()
parsed = __import__("json").loads(clean)
print(parsed["answer"])
```

### Output
```text
42
```

### Input (with RactoGateway)
```python
from ractogateway import RactoPrompt
from ractogateway.adapters.base import try_parse_json, strip_markdown_fences

prompt = RactoPrompt(
    role="You are a precise assistant.",
    aim="Return the answer.",
    constraints=["Return only what is asked."],
    tone="Concise",
    output_format="json",
)

raw_text = """```json
{"answer":"42"}
```"""
print(strip_markdown_fences(raw_text))
print(try_parse_json(raw_text))
```

### Output
```text
{"answer":"42"}
{'answer': '42'}
```

---

## Installation

### Input
```bash
# Core package
pip install ractogateway

# Provider extras
pip install ractogateway[openai]
pip install ractogateway[google]
pip install ractogateway[anthropic]

# All providers
pip install ractogateway[all]
```

### Output
```text
Successfully installed ractogateway-0.1.0
```

### Input (RAG extras)
```bash
# Minimal RAG stack
pip install ractogateway[rag]

# Full RAG stack (all vector stores + readers + voyage)
pip install ractogateway[rag-all]
```

### Output
```text
Successfully installed ractogateway-0.1.0 ... (RAG optional dependencies)
```

### Input (dev setup)
```bash
pip install -e ".[dev]"
```

### Output
```text
Successfully installed ractogateway-0.1.0 pytest ruff mypy ...
```

Requirements:
- Python `>=3.10`
- Pydantic `>=2.0,<3.0`

---

## Quick Start

### 1) Create a RACTO prompt

### Input
```python
from ractogateway import RactoPrompt

prompt = RactoPrompt(
    role="You are a senior Python reviewer.",
    aim="Find bugs and return JSON.",
    constraints=[
        "Do not guess.",
        "If no bug exists, return an empty issues list.",
    ],
    tone="Professional",
    output_format="json",
)

print(prompt.compile().splitlines()[:8])
```

### Output
```text
['[ROLE]', 'You are a senior Python reviewer.', '', '[AIM]', 'Find bugs and return JSON.', '', '[CONSTRAINTS]', '- Do not guess.']
```

### 2) Use a developer kit (OpenAI example)

### Input
```python
from ractogateway import openai_developer_kit as opd

kit = opd.OpenAIDeveloperKit(
    model="gpt-4o",
    api_key="sk-...",      # or OPENAI_API_KEY
    default_prompt=prompt,
)

cfg = opd.ChatConfig(user_message="Review: def add(a,b): return a+b")
resp = kit.chat(cfg)

print(resp.content)
print(resp.parsed)
print(resp.finish_reason)
print(resp.usage)
```

### Output (example)
```text
{"issues":[],"risk":"low","summary":"No bug found in provided snippet."}
{'issues': [], 'risk': 'low', 'summary': 'No bug found in provided snippet.'}
FinishReason.STOP
{'prompt_tokens': 112, 'completion_tokens': 23, 'total_tokens': 135}
```

### 3) Stream response chunks

### Input
```python
for chunk in kit.stream(opd.ChatConfig(user_message="Explain Python generators in 3 bullets.")):
    print(chunk.delta.text, end="", flush=True)
    if chunk.is_final:
        print("\nFINAL:", chunk.finish_reason, chunk.usage)
```

### Output (example)
```text
1) A generator yields values lazily.
2) It preserves state between yields.
3) It saves memory for large sequences.
FINAL: FinishReason.STOP {'prompt_tokens': 98, 'completion_tokens': 36, 'total_tokens': 134}
```

### 4) Async usage

### Input
```python
import asyncio

async def main():
    r = await kit.achat(opd.ChatConfig(user_message="What is SOLID in one sentence?"))
    print("achat:", r.content)

    async for c in kit.astream(opd.ChatConfig(user_message="Give SOLID as bullets.")):
        print(c.delta.text, end="")
    print()

asyncio.run(main())
```

### Output (example)
```text
achat: SOLID is a set of five design principles for maintainable OO software.
- S: Single Responsibility
- O: Open/Closed
- L: Liskov Substitution
- I: Interface Segregation
- D: Dependency Inversion
```

---

## Developer Kits

RactoGateway ships 3 kit modules with a consistent surface:
- `opd` -> OpenAI
- `god` -> Google Gemini
- `anth` -> Anthropic Claude

### Input
```python
from ractogateway import openai_developer_kit as opd
from ractogateway import google_developer_kit as god
from ractogateway import anthropic_developer_kit as anth

print(opd.OpenAIDeveloperKit.provider)
print(god.GoogleDeveloperKit.provider)
print(anth.AnthropicDeveloperKit.provider)
```

### Output
```text
openai
google
anthropic
```

### Method support matrix

| Method | OpenAI (`opd`) | Google (`god`) | Anthropic (`anth`) |
| --- | :---: | :---: | :---: |
| `chat(config)` | Yes | Yes | Yes |
| `achat(config)` | Yes | Yes | Yes |
| `stream(config)` | Yes | Yes | Yes |
| `astream(config)` | Yes | Yes | Yes |
| `embed(config)` | Yes | Yes | No |
| `aembed(config)` | Yes | Yes | No |

### Input (embedding with OpenAI)
```python
ecfg = opd.EmbeddingConfig(texts=["hello", "world"])
eresp = kit.embed(ecfg)
print(eresp.model)
print(len(eresp.vectors), len(eresp.vectors[0].embedding))
```

### Output (example)
```text
text-embedding-3-small
2 1536
```

---

## Input Models

All call inputs are Pydantic models.

### `ChatConfig`

### Input
```python
from ractogateway import openai_developer_kit as opd

cfg = opd.ChatConfig(
    user_message="Explain monads simply.",
    temperature=0.2,
    max_tokens=400,
    extra={"top_p": 0.9},
)
print(cfg.model_dump())
```

### Output
```text
{'user_message': 'Explain monads simply.', 'prompt': None, 'temperature': 0.2, 'max_tokens': 400, 'tools': None, 'response_model': None, 'history': [], 'extra': {'top_p': 0.9}}
```

### `EmbeddingConfig`

### Input
```python
ecfg = opd.EmbeddingConfig(
    texts=["first text", "second text"],
    model="text-embedding-3-small",
    dimensions=512,
)
print(ecfg.model_dump())
```

### Output
```text
{'texts': ['first text', 'second text'], 'model': 'text-embedding-3-small', 'dimensions': 512, 'extra': {}}
```

---

## Output Models

### `LLMResponse`

### Input
```python
r = resp
print(type(r.content).__name__)
print(type(r.parsed).__name__ if r.parsed is not None else None)
print(r.finish_reason)
print(r.usage)
```

### Output (example)
```text
str
dict
FinishReason.STOP
{'prompt_tokens': 112, 'completion_tokens': 23, 'total_tokens': 135}
```

### `StreamChunk`

### Input
```python
chunk = next(iter(kit.stream(opd.ChatConfig(user_message="Say hello."))))
print(chunk.delta.text)
print(chunk.accumulated_text)
print(chunk.is_final)
```

### Output (example)
```text
Hello
Hello
False
```

### `EmbeddingResponse`

### Input
```python
print(eresp.vectors[0].index)
print(eresp.vectors[0].text)
print(len(eresp.vectors[0].embedding))
```

### Output (example)
```text
0
hello
1536
```

---

## RACTO Prompt Engine

RACTO fields:
- `role`
- `aim`
- `constraints`
- `tone`
- `output_format`

### Input (`compile()`)
```python
print(prompt.compile())
```

### Output (example)
```text
[ROLE]
You are a senior Python reviewer.

[AIM]
Find bugs and return JSON.

[CONSTRAINTS]
- Do not guess.
- If no bug exists, return an empty issues list.

[TONE]
Professional

[OUTPUT]
Respond ONLY with valid JSON. Do NOT wrap the response in markdown code fences (```json ... ```) or add any commentary before or after the JSON object.

[GUARDRAILS]
- If you are unsure or lack sufficient information, state it explicitly rather than guessing.
- Do NOT fabricate facts, citations, URLs, statistics, or code that you cannot verify.
- Stick strictly to what is asked. Do not add unrequested information.
- If the answer requires assumptions, list each assumption explicitly before proceeding.
```

### Input (Pydantic model as output schema)
```python
from pydantic import BaseModel
from ractogateway import RactoPrompt

class ReviewResult(BaseModel):
    issues: list[str]
    risk: str
    summary: str

schema_prompt = RactoPrompt(
    role="You are a reviewer.",
    aim="Return structured review output.",
    constraints=["Only real issues."],
    tone="Concise",
    output_format=ReviewResult,
)

compiled = schema_prompt.compile()
print("JSON Schema:" in compiled)
print("issues" in compiled and "risk" in compiled and "summary" in compiled)
```

### Output
```text
True
True
```

### Input (optional fields)
```python
p2 = RactoPrompt(
    role="You are an analyst.",
    aim="Answer question.",
    constraints=["Be factual."],
    tone="Neutral",
    output_format="text",
    context="Company dataset: Q1-Q4 2025 KPIs.",
    examples=[{"input": "What is ARR?", "output": "Annual Recurring Revenue."}],
    anti_hallucination=True,
)
print("[CONTEXT]" in p2.compile(), "[EXAMPLES]" in p2.compile(), "[GUARDRAILS]" in p2.compile())
```

### Output
```text
True True True
```

---

## Multimodal Attachments (RactoFile)

Teaching note:
- `RactoFile` is fully supported in `prompt.to_messages(...)`.
- Fine-tuning datasets support multimodal examples.
- Current `ChatConfig` takes `user_message: str`, so for direct kit chat use text input.

### Input (`RactoFile.from_path` and `from_bytes`)
```python
from ractogateway.prompts.engine import RactoFile

png_bytes = b"\x89PNG\r\n\x1a\nFAKEPNGDATA"
txt_bytes = b"hello from text file"

img = RactoFile.from_bytes(png_bytes, "image/png", name="chart.png")
txt = RactoFile.from_bytes(txt_bytes, "text/plain", name="notes.txt")

print(img.mime_type, img.is_image, img.is_pdf, img.is_text)
print(txt.mime_type, txt.is_image, txt.is_pdf, txt.is_text)
```

### Output
```text
image/png True False False
text/plain False False True
```

### Input (OpenAI message conversion)
```python
msgs_openai = prompt.to_messages(
    "Describe this attachment.",
    attachments=[img, txt],
    provider="openai",
)
print(msgs_openai[0]["role"])
print(msgs_openai[1]["role"])
print(type(msgs_openai[1]["content"]).__name__)
print(msgs_openai[1]["content"][0]["type"], msgs_openai[1]["content"][-1]["type"])
```

### Output
```text
system
user
list
image_url text
```

### Input (Anthropic message conversion)
```python
msgs_anth = prompt.to_messages(
    "Summarize files.",
    attachments=[img, txt],
    provider="anthropic",
)
print(msgs_anth[1]["content"][0]["type"])
print(msgs_anth[1]["content"][1]["type"])
print(msgs_anth[1]["content"][-1]["type"])
```

### Output
```text
image
text
text
```

### Input (Google message conversion)
```python
msgs_google = prompt.to_messages(
    "What is in the file?",
    attachments=[img, txt],
    provider="google",
)
print("inline_data" in msgs_google[1]["content"][0])
print("text" in msgs_google[1]["content"][1])
```

### Output
```text
True
True
```

---

## Tool Calling

### Input (register tools)
```python
from ractogateway import ToolRegistry

registry = ToolRegistry()

@registry.register
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get weather by city."""
    return f"{city}: 22 degrees ({unit})"

print(len(registry))
print(registry.schemas[0].name)
print(registry.schemas[0].to_json_schema())
```

### Output
```text
1
get_weather
{'type': 'object', 'properties': {'city': {'type': 'string'}, 'unit': {'type': 'string', 'default': 'celsius'}}, 'required': ['city'], 'additionalProperties': False}
```

### Input (use tool registry in chat)
```python
cfg = opd.ChatConfig(
    user_message="What is the weather in Tokyo?",
    tools=registry,
)
r = kit.chat(cfg)

print("tool calls:", len(r.tool_calls))
for tc in r.tool_calls:
    fn = registry.get_callable(tc.name)
    result = fn(**tc.arguments) if fn else "missing tool"
    print(tc.name, tc.arguments, "=>", result)
```

### Output (example if model decides to call tool)
```text
tool calls: 1
get_weather {'city': 'Tokyo', 'unit': 'celsius'} => Tokyo: 22 degrees (celsius)
```

---

## Validated Response Models

### Input
```python
from pydantic import BaseModel

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    reasoning: str

cfg = opd.ChatConfig(
    user_message="Analyze sentiment: 'This product is amazing!'",
    response_model=SentimentResult,
)
r = kit.chat(cfg)
print(r.parsed)
```

### Output (example)
```text
{'sentiment': 'positive', 'confidence': 0.95, 'reasoning': "The adjective 'amazing' indicates strong positive sentiment."}
```

---

## Gateway (Unified Runner)

`Gateway` lets you drive any adapter from one runner class.

### Input
```python
from ractogateway import Gateway
from ractogateway.adapters.openai_kit import OpenAILLMKit

adapter = OpenAILLMKit(model="gpt-4o", api_key="sk-...")
gw = Gateway(adapter=adapter, default_prompt=prompt)

r = gw.run(user_message="Return one-line summary of Python decorators.")
print(r.content)
print(r.finish_reason)
```

### Output (example)
```text
Decorators are callables that wrap functions/classes to extend behavior without changing original source.
FinishReason.STOP
```

---

## Switching Providers (Same Pattern, Different Kit)

### Input
```python
from ractogateway import openai_developer_kit as opd
from ractogateway import google_developer_kit as god
from ractogateway import anthropic_developer_kit as anth

cfg = opd.ChatConfig(user_message="What is quantum computing in one sentence?")

okit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=prompt)
gkit = god.GoogleDeveloperKit(model="gemini-2.0-flash", default_prompt=prompt)
akit = anth.AnthropicDeveloperKit(model="claude-sonnet-4-5-20250929", default_prompt=prompt)

print("openai:", okit.chat(cfg).content[:60])
print("google:", gkit.chat(cfg).content[:60])
print("anthropic:", akit.chat(cfg).content[:60])
```

### Output (example)
```text
openai: Quantum computing uses qubits that can exist in superpositions
google: Quantum computing is a model of computation using qubits
anthropic: Quantum computing processes information using quantum states
```

---

## RAG (Retrieval-Augmented Generation)

This is a fully local teaching demo using:
- `InMemoryVectorStore`
- a tiny custom embedder
- a tiny mock LLM kit

No external API keys required for this demo.

### Input
```python
from ractogateway import RactoRAG, InMemoryVectorStore
from ractogateway.adapters.base import LLMResponse
from ractogateway.rag.embedders.base import BaseEmbedder

class ToyEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), float(sum(ord(c) for c in t) % 997)] for t in texts]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

class ToyLLMKit:
    def chat(self, config):
        return LLMResponse(content="Answer based on retrieved context: OpenAI, Google, Anthropic.")

    async def achat(self, config):
        return self.chat(config)

rag = RactoRAG(
    vector_store=InMemoryVectorStore(),
    embedder=ToyEmbedder(),
    llm_kit=ToyLLMKit(),
)

rag.ingest_text("RactoGateway unifies OpenAI, Google Gemini, and Anthropic Claude.", source="intro")
rag.ingest_text("It also supports tool calling, embeddings, and structured outputs.", source="features")

print("indexed chunks:", rag.count())

hits = rag.retrieve("Which providers are supported?", top_k=2)
for h in hits:
    print(h.rank, round(h.score, 4), h.chunk.metadata.source)

answer = rag.query("Which providers are supported?", top_k=2)
print(answer.answer.content)
print("sources used:", len(answer.sources))
```

### Output
```text
indexed chunks: 2
1 0.9999 intro
2 0.9987 features
Answer based on retrieved context: OpenAI, Google, Anthropic.
sources used: 2
```

---

## Fine-Tuning (Dataset + Tuners)

RactoGateway gives one dataset model for all providers.

### Step 1: Build dataset

### Input
```python
from ractogateway import RactoDataset

ds = RactoDataset.from_pairs(
    [
        ("What is a list?", "An ordered, mutable sequence."),
        ("What is a dict?", "A key-value mapping."),
        ("What is a tuple?", "An ordered, immutable sequence."),
    ],
    system="You are a concise Python tutor.",
)

print(ds.summary())
```

### Output
```text
{'examples': 3, 'total_messages': 9, 'avg_turns_per_example': 3.0, 'multimodal_examples': 0}
```

### Step 2: Validate and split

### Input
```python
errors = ds.validate("openai")
print(errors)

train_ds, val_ds = ds.split(train_ratio=0.67, seed=42)
print("train:", len(train_ds), "val:", len(val_ds))
```

### Output
```text
[]
train: 2 val: 1
```

### Step 3: Export JSONL

### Input
```python
train_ds.export_jsonl("train_openai.jsonl", provider="openai", overwrite=True)
train_ds.export_jsonl("train_anthropic.jsonl", provider="anthropic", overwrite=True)
train_ds.export_jsonl("train_gemini.jsonl", provider="gemini", overwrite=True)

print(train_ds.to_jsonl_string("openai").splitlines()[0])
print(train_ds.to_jsonl_string("anthropic").splitlines()[0])
print(train_ds.to_jsonl_string("gemini").splitlines()[0])
```

### Output (example)
```text
{"messages": [{"role": "system", "content": "You are a concise Python tutor."}, {"role": "user", "content": "What is a tuple?"}, {"role": "assistant", "content": "An ordered, immutable sequence."}]}
{"messages": [{"role": "user", "content": "What is a tuple?"}, {"role": "assistant", "content": "An ordered, immutable sequence."}], "system": "You are a concise Python tutor."}
{"text_input": "What is a tuple?", "output": "An ordered, immutable sequence."}
```

### Step 4: Run provider tuner (requires API key access)

### Input (OpenAI one-call pipeline)
```python
from ractogateway import OpenAIFineTuner

tuner = OpenAIFineTuner(api_key="sk-...")
fine_tuned_model = tuner.run_pipeline(
    train_ds,
    model="gpt-4o-mini-2024-07-18",
    validation_dataset=val_ds,
    n_epochs=3,
    suffix="python-tutor",
    verbose=True,
)
print("model:", fine_tuned_model)
```

### Output (example)
```text
[OpenAIFineTuner] Uploading 2 training examples (0 multimodal)...
[OpenAIFineTuner] Training file: file-abc123
[OpenAIFineTuner] Uploading 1 validation examples...
[OpenAIFineTuner] Validation file: file-val123
[OpenAIFineTuner] Job created: ftjob-xyz789
[OpenAIFineTuner] Job ftjob-xyz789 -> running
[OpenAIFineTuner] Job ftjob-xyz789 -> succeeded
[OpenAIFineTuner] Done!  Fine-tuned model: ft:gpt-4o-mini-2024-07-18:org::python-tutor-abc
model: ft:gpt-4o-mini-2024-07-18:org::python-tutor-abc
```

### Input (Gemini one-call pipeline, text pairs only)
```python
from ractogateway import GeminiFineTuner

gtuner = GeminiFineTuner(api_key="AIza...")
tuned_model = gtuner.run_pipeline(
    train_ds,
    base_model="models/gemini-1.5-flash-001-tuning",
    display_name="python-tutor",
    epoch_count=5,
    batch_size=4,
    verbose=True,
)
print("model:", tuned_model)
```

### Output (example)
```text
[GeminiFineTuner] Starting tuning with 2 examples...
[GeminiFineTuner] State: CREATING (15%)
[GeminiFineTuner] State: RUNNING (80%)
[GeminiFineTuner] Done!  Tuned model: tunedModels/python-tutor-abc123
model: tunedModels/python-tutor-abc123
```

### Input (Anthropic one-call pipeline)
```python
from ractogateway import AnthropicFineTuner

atuner = AnthropicFineTuner(api_key="sk-ant-...")
amodel = atuner.run_pipeline(
    train_ds,
    model="claude-3-haiku-20240307",
    validation_dataset=val_ds,
    suffix="python-tutor",
    hyperparameters={"n_epochs": 3},
    verbose=True,
)
print("model:", amodel)
```

### Output (example)
```text
[AnthropicFineTuner] Uploading 2 training examples (0 multimodal)...
[AnthropicFineTuner] Training file: file-abc
[AnthropicFineTuner] Uploading 1 validation examples...
[AnthropicFineTuner] Validation file: file-val
[AnthropicFineTuner] Job created: ftjob-123
[AnthropicFineTuner] Job ftjob-123 -> running
[AnthropicFineTuner] Job ftjob-123 -> completed
[AnthropicFineTuner] Done!  Fine-tuned model: claude-3-haiku-20240307:ft:org:suffix:abc
model: claude-3-haiku-20240307:ft:org:suffix:abc
```

---

## Environment Variables

### Input
```python
import os
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
print("GEMINI_API_KEY set:", bool(os.getenv("GEMINI_API_KEY")))
print("ANTHROPIC_API_KEY set:", bool(os.getenv("ANTHROPIC_API_KEY")))
```

### Output (example)
```text
OPENAI_API_KEY set: True
GEMINI_API_KEY set: False
ANTHROPIC_API_KEY set: True
```

Reference:
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`

---

## Architecture

### Input
```text
src/ractogateway/
```

### Output
```text
src/ractogateway/
|-- __init__.py
|-- _models/
|   |-- chat.py
|   |-- stream.py
|   `-- embedding.py
|-- prompts/
|   `-- engine.py
|-- tools/
|   `-- registry.py
|-- adapters/
|   |-- base.py
|   |-- openai_kit.py
|   |-- google_kit.py
|   `-- anthropic_kit.py
|-- gateway/
|   `-- runner.py
|-- openai_developer_kit/
|   `-- kit.py
|-- google_developer_kit/
|   `-- kit.py
|-- anthropic_developer_kit/
|   `-- kit.py
|-- rag/
|   |-- pipeline.py
|   |-- readers/
|   |-- chunkers/
|   |-- embedders/
|   |-- processors/
|   `-- stores/
`-- finetune/
    |-- dataset.py
    |-- openai_tuner.py
    |-- gemini_tuner.py
    `-- anthropic_tuner.py
```

---

## Contributing

### Input
```bash
git clone https://github.com/IAMPathak2702/RactoGateway.git
cd RactoGateway
pip install -e ".[dev]"

pytest
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

### Output (example)
```text
============================= test session starts =============================
collected N items
...
============================== N passed in 0.XXs ==============================

All checks passed!
```

---

## License

### Input
```text
License type
```

### Output
```text
Apache License 2.0
```

See `LICENSE` for details.

---

## Author

### Input
```text
Project maintainer
```

### Output
```text
Ved Prakash Pathak
GitHub: @IAMPathak2702
Email: vp.ved.vpp@gmail.com
```
