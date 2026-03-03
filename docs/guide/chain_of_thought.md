# Chain of Thoughts

Set `chain_of_thought=True` on any `ChatConfig` to make the model reason step by step
before answering. The feature works identically across all five provider kits —
OpenAI, Anthropic, Google, Ollama, and HuggingFace — with zero adapter changes.

## How it works

When `chain_of_thought=True`, the kit creates a non-mutating copy of the active
`RactoPrompt` and appends the following constraint to the `[CONSTRAINTS]` section of
the compiled system prompt:

> *"Before answering, reason through the problem step by step. State each reasoning
> step clearly and explicitly, then conclude with your final answer."*

The constraint is added **last**, after all caller-defined rules, so it never
overrides your own constraints. The original `RactoPrompt` object is never mutated.

## Basic usage

```python
from ractogateway import openai_developer_kit as gpt, RactoPrompt

prompt = RactoPrompt(
    role="You are a maths tutor.",
    aim="Solve the problem the student gives you.",
    constraints=["Show every calculation step.", "Use plain English."],
    tone="Patient and encouraging",
    output_format="text",
)
kit = gpt.Chat(model="gpt-4o", default_prompt=prompt)

response = kit.chat(gpt.ChatConfig(
    user_message="What is 17 × 23?",
    chain_of_thought=True,
))
print(response.content)
# "Step 1: Break 17 × 23 into (17 × 20) + (17 × 3).
#  Step 2: 17 × 20 = 340.
#  Step 3: 17 × 3 = 51.
#  Step 4: 340 + 51 = 391.
#  Answer: 391."
```

## Works with every kit

Swap `gpt` for any other kit — the API is identical:

```python
from ractogateway import anthropic_developer_kit as claude
from ractogateway import google_developer_kit as gemini
from ractogateway import ollama_developer_kit as local
from ractogateway import huggingface_developer_kit as hf

config = claude.ChatConfig(user_message="Explain recursion.", chain_of_thought=True)
response = claude.Chat(model="claude-opus-4-6", default_prompt=prompt).chat(config)
```

## Works with streaming

```python
for chunk in kit.stream(gpt.ChatConfig(
    user_message="Explain the travelling salesman problem.",
    chain_of_thought=True,
    temperature=0.4,
)):
    print(chunk.delta.text, end="", flush=True)
```

## Tips

| Goal | Setting |
| --- | --- |
| Richer, more expressive reasoning | Raise `temperature` to `0.3`–`0.7` |
| Long multi-step reasoning chains | Increase `max_tokens` (default `4096`) |
| Structured final answer only | Add `output_format=YourModel` to the prompt; CoT reasons internally then outputs JSON |
| Disable for a specific call | Simply omit `chain_of_thought` (default `False`) |

## `ChatConfig` field reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chain_of_thought` | `bool` | `False` | Inject CoT reasoning instruction |
| `temperature` | `float` | `0.0` | Raise for richer reasoning traces |
| `max_tokens` | `int` | `4096` | Increase for long reasoning chains |

## Implementation details

The helper lives in `src/ractogateway/_cot.py`:

```python
from ractogateway._cot import apply_chain_of_thought

modified_prompt = apply_chain_of_thought(original_prompt)
# Returns a new RactoPrompt; original_prompt is unchanged
```

Each kit calls `apply_chain_of_thought` immediately after resolving the active prompt
(in `chat()`, `achat()`, `stream()`, and `astream()`), so the constraint is always
present in the compiled system prompt that the adapter receives.
