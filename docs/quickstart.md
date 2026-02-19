# Quick Start

```python
from ractogateway.openai_developer_kit import OpenAIDeveloperKit
from ractogateway.prompts.engine import RactoPrompt
from pydantic import BaseModel

class Answer(BaseModel):
    text: str
    confidence: float

kit = OpenAIDeveloperKit(api_key="sk-...")
prompt = RactoPrompt(
    role="You are a helpful assistant.",
    task="Answer the question.",
    context="The sky is blue because of Rayleigh scattering.",
    output_format=Answer,
)
result = kit.complete.sync(prompt)
print(result)  # Answer(text=..., confidence=...)
```

See the [User Guide](guide/developer_kits.md) for full examples.
