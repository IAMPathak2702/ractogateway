# Native Thinking

Set `native_thinking=True` on any `ChatConfig` to get the model's **actual internal
reasoning process** — not just a prompted instruction to "think step by step", but the
real tokens the model spends working through the problem before writing its answer.

This is different from `chain_of_thought=True` (which is a prompt engineering trick that
works on all models). Native thinking is a **model-level feature** supported only by
specific models, and the thinking text appears in dedicated fields separated from the
final answer.

## Supported models

| Provider | Models | Thinking visible? |
| --- | --- | --- |
| **Anthropic** | `claude-3-7-sonnet-20250219` and later | Yes — full text, streamable |
| **Google** | `gemini-2.5-pro`, `gemini-2.0-flash-thinking-exp` | Yes — full text, streamable |
| **OpenAI** | `o1`, `o3`, `o3-mini` | No — token count only (`usage["reasoning_tokens"]`) |

---

## Quick start — Anthropic (streaming)

```python
from ractogateway import anthropic_developer_kit as claude
from ractogateway.prompts.engine import RactoPrompt

prompt = RactoPrompt(
    role="Expert mathematician",
    aim="Solve the user's problem completely.",
    constraints=["Show the final answer clearly."],
    tone="Precise",
    output_format="text",
)
kit = claude.Chat(
    model="claude-3-7-sonnet-20250219",
    default_prompt=prompt,
)

print("=== THINKING ===")
for chunk in kit.stream(claude.ChatConfig(
    user_message="How many trailing zeros does 100! have?",
    native_thinking=True,
    thinking_budget=8000,
)):
    if chunk.is_thinking:
        print(chunk.delta.thinking, end="", flush=True)
    elif chunk.delta.text:
        if not chunk.accumulated_thinking:
            print("\n=== ANSWER ===")
        print(chunk.delta.text, end="", flush=True)
```

**Expected output:**

```
=== THINKING ===
I need to find the number of trailing zeros in 100!.
Trailing zeros come from factors of 10, and 10 = 2 × 5.
Since factors of 2 appear much more often than 5, I just need to count factors of 5.

Factors of 5 in 100!:
- Numbers divisible by 5:  100 ÷ 5  = 20
- Numbers divisible by 25: 100 ÷ 25 = 4
- Numbers divisible by 125: 100 ÷ 125 = 0 (125 > 100)

Total = 20 + 4 = 24

=== ANSWER ===
100! has **24 trailing zeros**.
```

---

## Quick start — Anthropic (non-streaming)

```python
response = kit.chat(claude.ChatConfig(
    user_message="How many trailing zeros does 100! have?",
    native_thinking=True,
    thinking_budget=8000,
))

print("THINKING:", response.thinking)
# THINKING: I need to find the number of trailing zeros in 100!.
# Trailing zeros come from factors of 10, and 10 = 2 × 5. ...

print("ANSWER:", response.content)
# ANSWER: 100! has **24 trailing zeros**.
```

---

## Quick start — Google Gemini (streaming)

```python
from ractogateway import google_developer_kit as gemini

kit = gemini.Chat(
    model="gemini-2.5-pro",
    default_prompt=prompt,
)

for chunk in kit.stream(gemini.ChatConfig(
    user_message="What is the probability of getting at least one 6 in four dice rolls?",
    native_thinking=True,
    thinking_budget=4096,
)):
    if chunk.is_thinking:
        print(chunk.delta.thinking, end="", flush=True)
    elif chunk.delta.text:
        print(chunk.delta.text, end="", flush=True)
```

**Expected output (thinking):**

```
The complement of "at least one 6" is "no 6 in any of the four rolls".
P(no 6 on a single roll) = 5/6
P(no 6 in four rolls)    = (5/6)^4 = 625/1296
P(at least one 6)        = 1 − 625/1296 = 671/1296 ≈ 0.5177
```

**Expected output (answer):**

```
The probability is **671/1296 ≈ 51.8%**.
```

---

## Quick start — OpenAI o-series (reasoning token count)

OpenAI o1/o3 models reason internally and do not expose the reasoning text.
`native_thinking=True` is accepted but has no API effect — the reasoning token count
is always added to `usage["reasoning_tokens"]` automatically when available.

```python
from ractogateway import openai_developer_kit as gpt

kit = gpt.Chat(model="o3-mini", default_prompt=prompt)

response = kit.chat(gpt.ChatConfig(
    user_message="Solve x² − 5x + 6 = 0",
    native_thinking=True,   # optional flag; no-op for OpenAI
))

print(response.content)
# x = 2  or  x = 3

print(response.usage)
# {
#   "prompt_tokens": 142,
#   "completion_tokens": 89,
#   "total_tokens": 231,
#   "reasoning_tokens": 64      ← reasoning tokens consumed
# }
```

---

## Controlling the thinking budget

```python
config = claude.ChatConfig(
    user_message="Explain why P ≠ NP is hard to prove.",
    native_thinking=True,
    thinking_budget=20000,   # more budget → deeper reasoning
    max_tokens=8192,         # must be > thinking_budget for Anthropic
)
```

| Parameter | Default | Notes |
| --- | --- | --- |
| `thinking_budget` | `10000` | Max tokens the model may spend reasoning |
| `max_tokens` | `4096` | Anthropic: must be set higher than the budget |

> **Anthropic note:** `temperature` is automatically forced to `1` — you do not need
> to set it. Passing any other value is silently overridden.

---

## Reading the output

### Streaming

```python
for chunk in kit.stream(config):
    # --- thinking phase ---
    chunk.is_thinking            # True while thinking tokens stream
    chunk.delta.thinking         # the new reasoning text in this event
    chunk.accumulated_thinking   # all reasoning text so far

    # --- answer phase ---
    chunk.delta.text             # the new answer text in this event
    chunk.accumulated_text       # all answer text so far

    # --- final chunk ---
    chunk.is_final               # True on the last event
    chunk.accumulated_thinking   # complete reasoning
    chunk.accumulated_text       # complete answer
    chunk.usage                  # token counts
```

### Non-streaming

```python
response = kit.chat(config)

response.thinking   # str | None  — complete reasoning text
response.content    # str | None  — final answer
response.usage      # dict with prompt_tokens, completion_tokens, total_tokens
                    # (+ reasoning_tokens for OpenAI o-series)
```

---

## Combining with `chain_of_thought`

`native_thinking` and `chain_of_thought` can be used together.
`chain_of_thought` adds a prompt-level instruction; `native_thinking` activates the
model's internal engine. On Anthropic/Google the model will reason both internally
(native) and then also tend to be more explicit in its answer (prompted).

```python
config = claude.ChatConfig(
    user_message="...",
    native_thinking=True,
    chain_of_thought=True,
)
```

---

## Practical: render thinking in a terminal

```python
import sys

previous_was_thinking = False

for chunk in kit.stream(config):
    if chunk.is_thinking:
        if not previous_was_thinking:
            print("\033[2m[thinking]\033[0m", flush=True)   # dim
        print(f"\033[2m{chunk.delta.thinking}\033[0m", end="", flush=True)
        previous_was_thinking = True
    else:
        if previous_was_thinking:
            print("\n\033[0m[answer]\033[0m", flush=True)   # reset
            previous_was_thinking = False
        print(chunk.delta.text, end="", flush=True)

print()
```

---

## Tips

| Goal | Setting |
| --- | --- |
| Deep, multi-step proofs / code | Raise `thinking_budget` to `32000`–`100000` |
| Fast simple answers | Lower `thinking_budget` to `1024`–`2048` |
| Only show the final answer | Ignore `chunk.delta.thinking`; read `chunk.accumulated_text` |
| Log full reasoning for debugging | Save `chunk.accumulated_thinking` on `chunk.is_final` |
| Async streaming | Use `astream()` — same fields, same behaviour |
