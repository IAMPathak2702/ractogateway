# HuggingFace — Cloud and Local Inference

`HuggingFaceDeveloperKit` supports three deployment modes through a single,
consistent interface:

| Mode | When to use |
| --- | --- |
| **HuggingFace Inference API** | Quick prototyping; no server required; set `HF_TOKEN` |
| **Local TGI** | Self-hosted [Text Generation Inference](https://github.com/huggingface/text-generation-inference); no API key needed |
| **Local vLLM / Llama.cpp** | Any OpenAI-compatible HTTP server; pass `base_url` |

## Installation

```bash
pip install ractogateway[huggingface]
```

For cloud inference, obtain a token at <https://huggingface.co/settings/tokens>
and set:

```bash
export HF_TOKEN="hf_..."
```

## Quick Start — Cloud Inference

```python
from ractogateway import huggingface_developer_kit as hf, RactoPrompt

prompt = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Answer the user clearly and concisely.",
    constraints=["Stay on topic.", "Do not hallucinate."],
    tone="Friendly",
    output_format="text",
)

# Token read from HF_TOKEN env var automatically
kit = hf.Chat(
    model="meta-llama/Llama-3.2-3B-Instruct",
    default_prompt=prompt,
)

response = kit.chat(hf.ChatConfig(user_message="What is attention in transformers?"))
print(response.content)
```

## Quick Start — Local TGI Server

```bash
# Pull and launch TGI (requires Docker + enough VRAM/RAM)
docker run --rm -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference \
  --model-id meta-llama/Llama-3.2-3B-Instruct
```

```python
# No API key; point base_url at the running container
kit = hf.Chat(
    model="tgi",
    base_url="http://localhost:8080",
    default_prompt=prompt,
)

response = kit.chat(hf.ChatConfig(user_message="Explain attention in one paragraph."))
print(response.content)
```

## Quick Start — Local vLLM Server

```bash
# Launch vLLM (OpenAI-compatible endpoint)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --port 8000
```

```python
kit = hf.Chat(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8000/v1",
    default_prompt=prompt,
)
```

## Constructor Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `str` | `"meta-llama/Llama-3.2-3B-Instruct"` | HF repo ID or server label |
| `api_key` | `str \| None` | `None` | Falls back to `HF_TOKEN` / `HUGGINGFACE_TOKEN` |
| `base_url` | `str \| None` | `None` | Local server URL (TGI, vLLM, Llama.cpp) |
| `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Default embedding model |
| `default_prompt` | `RactoPrompt \| None` | `None` | Kit-level default prompt |
| `exact_cache` | `ExactMatchCache \| None` | `None` | In-process exact-match cache |
| `semantic_cache` | `SemanticCache \| None` | `None` | Cosine-similarity cache |
| `router` | `CostAwareRouter \| None` | `None` | Required when `model="auto"` |
| `truncator` | `TokenTruncator \| None` | `None` | Auto-trim long histories |
| `tracer` | `RactoTracer \| None` | `None` | OpenTelemetry spans |
| `metrics` | `GatewayMetricsMiddleware \| None` | `None` | Prometheus metrics |

## Streaming

```python
for chunk in kit.stream(hf.ChatConfig(user_message="Tell me a short story.")):
    print(chunk.delta.text, end="", flush=True)
    if chunk.is_final:
        print(f"\nTokens: {chunk.usage}")
```

## Async

```python
import asyncio

async def main() -> None:
    response = await kit.achat(hf.ChatConfig(user_message="Explain async/await."))
    print(response.content)

    async for chunk in kit.astream(hf.ChatConfig(user_message="Count to five.")):
        print(chunk.delta.text, end="", flush=True)

asyncio.run(main())
```

## Embeddings

`HuggingFaceDeveloperKit` uses `InferenceClient.feature_extraction()` for
embeddings. The model must support sentence / feature extraction (e.g.
`sentence-transformers/all-MiniLM-L6-v2`).

```python
embed_kit = hf.Chat(
    model="meta-llama/Llama-3.2-3B-Instruct",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    default_prompt=prompt,
)

resp = embed_kit.embed(hf.EmbeddingConfig(texts=["hello world", "goodbye world"]))
for vec in resp.vectors:
    print(f"[{vec.index}] '{vec.text}' — dim={len(vec.embedding)}")
```

## Vision Models (Image Input)

HuggingFace models that support the OpenAI-compatible ``image_url`` content
block format (e.g. ``llava-hf/llava-1.5-7b-hf``,
``Qwen/Qwen2-VL-7B-Instruct``) accept image attachments via
``ChatConfig.attachments``:

```python
from ractogateway.prompts.engine import RactoFile

img = RactoFile.from_path("/tmp/chart.png")

kit = hf.Chat(
    model="llava-hf/llava-1.5-7b-hf",
    default_prompt=prompt,
)
response = kit.chat(
    hf.ChatConfig(
        user_message="What trend do you see in this chart?",
        attachments=[img],
    )
)
print(response.content)
```

For local TGI deployments, enable multimodal support when launching the
container:

```bash
docker run --rm -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference \
  --model-id llava-hf/llava-1.5-7b-hf
```

## Tool Calling

Tool calling works on any model that supports function calling through the
HuggingFace chat completions API.

```python
from ractogateway import ToolRegistry, tool

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 22 °C in {city}"

registry = ToolRegistry([get_weather])

response = kit.chat(
    hf.ChatConfig(
        user_message="What is the weather in London?",
        tools=registry,
        auto_execute_tools=True,
    )
)
print(response.content)
```

## Validated JSON Output

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str   # "positive" | "negative" | "neutral"
    score: float

typed_prompt = RactoPrompt(
    role="You are a sentiment analyser.",
    aim="Classify the sentiment of the text.",
    constraints=["Return only the JSON object."],
    tone="Neutral",
    output_format=Sentiment,
)

kit = hf.Chat(
    model="meta-llama/Llama-3.2-3B-Instruct",
    default_prompt=typed_prompt,
)
response = kit.chat(
    hf.ChatConfig(
        user_message="I love this library!",
        response_model=Sentiment,
    )
)
print(response.parsed)   # validated Sentiment instance dict
```

## Environment Variables

| Variable | Used for |
| --- | --- |
| `HF_TOKEN` | HuggingFace Inference API authentication (preferred) |
| `HUGGINGFACE_TOKEN` | Alternative token env var name |

Both variables are checked in order. If neither is set and no `api_key` is
passed, the client will attempt unauthenticated access (works for some public
models but may be rate-limited).

## Recommended Models

| Use case | Model |
| --- | --- |
| General chat | `meta-llama/Llama-3.2-3B-Instruct` |
| Code generation | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Small + fast | `microsoft/Phi-3.5-mini-instruct` |
| Multilingual | `mistralai/Mistral-7B-Instruct-v0.3` |
| Embeddings (cloud) | `sentence-transformers/all-MiniLM-L6-v2` |
| Embeddings (local TGI) | Any `sentence-transformers` model |

Browse all models at <https://huggingface.co/models>.
