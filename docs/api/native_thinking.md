# Native Thinking API Reference

## ChatConfig fields

```{eval-rst}
.. autoclass:: ractogateway._models.chat.ChatConfig
   :members: native_thinking, thinking_budget
   :noindex:
```

---

## StreamDelta

```{eval-rst}
.. autoclass:: ractogateway._models.stream.StreamDelta
   :members:
   :noindex:
```

| Field | Type | Description |
| --- | --- | --- |
| `text` | `str` | Answer token delta (same as always) |
| `thinking` | `str` | Reasoning token delta (non-empty only on thinking chunks) |

---

## StreamChunk

```{eval-rst}
.. autoclass:: ractogateway._models.stream.StreamChunk
   :members:
   :noindex:
```

| Field | Type | Description |
| --- | --- | --- |
| `delta` | `StreamDelta` | Incremental content (`.text` or `.thinking`) |
| `accumulated_text` | `str` | All answer text received so far |
| `accumulated_thinking` | `str` | All reasoning text received so far |
| `is_thinking` | `bool` | `True` when this chunk carries only reasoning text |
| `is_final` | `bool` | `True` on the last event in the stream |
| `usage` | `dict[str, int]` | Token counts on the final chunk |

---

## LLMResponse

```{eval-rst}
.. autoclass:: ractogateway.adapters.base.LLMResponse
   :members: thinking
   :noindex:
```

| Field | Type | Description |
| --- | --- | --- |
| `content` | `str \| None` | Final answer text |
| `thinking` | `str \| None` | Complete reasoning text (Anthropic / Google) |
| `usage` | `dict[str, int]` | Token counts; OpenAI o-series adds `reasoning_tokens` key |

---

## Provider behaviour summary

### Anthropic

- Supported models: `claude-3-7-sonnet-20250219` and later.
- API param injected: `thinking={"type": "enabled", "budget_tokens": N}`.
- Temperature is automatically forced to `1` (API requirement; user value is ignored).
- Stream events: `thinking_delta` events carry `delta.thinking`; `text_delta` events
  carry `delta.text`. Both can interleave in the same stream.
- Non-streaming: `LLMResponse.thinking` contains the complete reasoning block.

### Google

- Supported models: `gemini-2.5-pro`, `gemini-2.0-flash-thinking-exp`.
- API param injected: `ThinkingConfig(thinking_budget=N)` in `GenerateContentConfig`.
- Thought parts are identified by `part.thought == True` in the response candidates.
- Streaming: thought parts arrive as `is_thinking=True` chunks before answer parts.
- Non-streaming: `LLMResponse.thinking` contains all joined thought parts.

### OpenAI

- Supported models: `o1`, `o3`, `o3-mini`.
- `native_thinking` / `thinking_budget` are silently ignored (OpenAI does not expose
  reasoning text in the Chat Completions API).
- Reasoning token count is exposed automatically in
  `response.usage["reasoning_tokens"]` whenever the model returns
  `completion_tokens_details.reasoning_tokens`.
