# Token Truncation

`TokenTruncator` prevents context-window overflow by trimming conversation history while preserving the system prompt and most-recent turns.

## Basic Usage

```python
from ractogateway.truncation import TokenTruncator, TruncationConfig
from ractogateway import openai_developer_kit as gpt

truncator = TokenTruncator(TruncationConfig(
    keep_first_n=2,      # always keep the first N messages (e.g. system prompt)
    keep_last_n=8,       # always keep the last N messages
    safety_margin=512,   # reserve tokens for the model's response
))

kit = gpt.OpenAIDeveloperKit(model="gpt-4o", truncator=truncator)
```

The truncator applies automatically before every API call.

## Precise Token Counting (OpenAI)

By default `TokenTruncator` estimates token count as `len(text) // 4`.  For exact counts install `tiktoken`:

```bash
pip install ractogateway[cache]   # includes tiktoken
```

```python
import tiktoken
from ractogateway.truncation import TokenTruncator, TruncationConfig

enc = tiktoken.encoding_for_model("gpt-4o")
truncator = TokenTruncator(TruncationConfig(
    token_counter=lambda t: len(enc.encode(t)),
    keep_last_n=10,
    safety_margin=1024,
))
```

## Model Context Limits

`MODEL_CONTEXT_LIMITS` is a pre-populated dict mapping common model names to their context window sizes:

```python
from ractogateway.truncation import MODEL_CONTEXT_LIMITS

print(MODEL_CONTEXT_LIMITS["gpt-4o"])        # 128000
print(MODEL_CONTEXT_LIMITS["claude-3-5-sonnet-20241022"])  # 200000
```
