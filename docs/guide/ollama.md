# Ollama — Local Model Inference

`OllamaDeveloperKit` lets you run any open-source LLM on your own hardware with
zero API key and zero data leaving your machine. It connects to a locally-running
[Ollama](https://ollama.com/) server and exposes the same six-method interface as
every other RactoGateway kit.

## Installation

```bash
# 1. Install Ollama  →  https://ollama.com/download
# 2. Pull any model
ollama pull llama3.2          # 2 GB — great for everyday tasks
ollama pull mistral           # 4 GB — excellent instruction following
ollama pull qwen2.5:7b        # 4.5 GB — strong multilingual model
ollama pull nomic-embed-text  # lightweight embeddings model

# 3. Install the Python extra
pip install ractogateway[ollama]
```

Ollama starts automatically on most platforms. If not, run:

```bash
ollama serve
```

## Quick Start

```python
from ractogateway import ollama_developer_kit as local, RactoPrompt

prompt = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Answer the user clearly and concisely.",
    constraints=["Stay on topic.", "Do not hallucinate."],
    tone="Friendly",
    output_format="text",
)

# No API key — Ollama listens at http://localhost:11434 by default
kit = local.Chat(model="llama3.2", default_prompt=prompt)

response = kit.chat(local.ChatConfig(user_message="What is a transformer model?"))
print(response.content)
```

## Constructor Parameters

```{eval-rst}
.. autoclass:: ractogateway.ollama_developer_kit.kit.OllamaDeveloperKit
   :members:
   :no-index:
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `str` | `"llama3.2"` | Model name from `ollama list` |
| `base_url` | `str` | `"http://localhost:11434"` | Ollama server base URL |
| `embedding_model` | `str` | `"nomic-embed-text"` | Default embedding model |
| `default_prompt` | `RactoPrompt \| None` | `None` | Kit-level default prompt |
| `exact_cache` | `ExactMatchCache \| None` | `None` | In-process exact-match cache |
| `semantic_cache` | `SemanticCache \| None` | `None` | Cosine-similarity cache |
| `router` | `CostAwareRouter \| None` | `None` | Required when `model="auto"` |
| `truncator` | `TokenTruncator \| None` | `None` | Auto-trim long histories |
| `tracer` | `RactoTracer \| None` | `None` | OpenTelemetry spans |
| `metrics` | `GatewayMetricsMiddleware \| None` | `None` | Prometheus metrics |

## Streaming

```python
for chunk in kit.stream(local.ChatConfig(user_message="Write a haiku about Python.")):
    print(chunk.delta.text, end="", flush=True)
    if chunk.is_final:
        print(f"\nTokens: {chunk.usage}")
```

## Async

```python
import asyncio

async def main() -> None:
    response = await kit.achat(local.ChatConfig(user_message="Explain async/await."))
    print(response.content)

    async for chunk in kit.astream(local.ChatConfig(user_message="Count to five.")):
        print(chunk.delta.text, end="", flush=True)

asyncio.run(main())
```

## Embeddings

Ollama requires a dedicated embedding model. Pull it first:

```bash
ollama pull nomic-embed-text
```

```python
embed_kit = local.Chat(
    model="llama3.2",
    embedding_model="nomic-embed-text",
    default_prompt=prompt,
)

resp = embed_kit.embed(local.EmbeddingConfig(texts=["hello world", "goodbye world"]))
for vec in resp.vectors:
    print(f"[{vec.index}] '{vec.text}' — dim={len(vec.embedding)}")
```

## Tool Calling

Ollama supports function calling on models that were trained with tool support
(e.g. `llama3.1`, `llama3.2`, `mistral-nemo`).

```python
from ractogateway import ToolRegistry, tool

@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"Sunny, 22 °C in {city}"

registry = ToolRegistry([get_weather])

response = kit.chat(
    local.ChatConfig(
        user_message="What's the weather in Paris?",
        tools=registry,
        auto_execute_tools=True,
    )
)
print(response.content)
```

## Pointing at a Remote Ollama Server

If Ollama runs on another machine (e.g. a GPU box), pass its address:

```python
kit = local.Chat(
    model="llama3.2",
    base_url="http://192.168.1.42:11434",
    default_prompt=prompt,
)
```

## Validated JSON Output

```python
from pydantic import BaseModel

class Summary(BaseModel):
    key_points: list[str]
    sentiment: str

typed_prompt = RactoPrompt(
    role="You are a text analyser.",
    aim="Summarise the text.",
    constraints=["Return only the JSON."],
    tone="Neutral",
    output_format=Summary,
)

kit = local.Chat(model="llama3.2", default_prompt=typed_prompt)
response = kit.chat(
    local.ChatConfig(
        user_message="Python is great for AI and scripting.",
        response_model=Summary,
    )
)
print(response.parsed)   # validated Summary instance dict
```

## Recommended Models

| Use case | Model | Size |
| --- | --- | --- |
| General chat | `llama3.2` | 2 GB |
| Code generation | `deepseek-coder-v2` | 9 GB |
| Long context | `llama3.1:8b` | 4.7 GB |
| Multilingual | `qwen2.5:7b` | 4.5 GB |
| Instruction following | `mistral` | 4 GB |
| Embeddings | `nomic-embed-text` | 274 MB |
| Small + fast | `phi3.5` | 2.2 GB |

Browse all models at <https://ollama.com/library>.
