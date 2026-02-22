# Batch Processing

Submit thousands of non-urgent LLM requests through provider Batch APIs at approximately 50 % lower cost than synchronous calls.

## OpenAI Batch Processor

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.prompts.engine import RactoPrompt

my_prompt = RactoPrompt(
    system="You are a helpful assistant.",
    output_format="Return a plain-text answer.",
)

processor = gpt.OpenAIBatchProcessor(model="gpt-4o-mini", default_prompt=my_prompt)

results = processor.submit_and_wait([
    gpt.BatchItem(custom_id="q1", user_message="Summarise this text: …"),
    gpt.BatchItem(custom_id="q2", user_message="Classify the sentiment: …"),
])

for r in results:
    if r.ok:
        print(r.custom_id, r.response.content)
    else:
        print(r.custom_id, "failed:", r.error)
```

## Anthropic Batch Processor

```python
from ractogateway import anthropic_developer_kit as claude

processor = claude.AnthropicBatchProcessor(
    model="claude-haiku-4-5-20251001",
    default_prompt=my_prompt,
)

results = processor.submit_and_wait(items)
```

## Checking Status

Both processors expose non-blocking status checks:

```python
job = processor.submit(items)
status = processor.get_status(job.id)   # BatchStatus.PENDING / IN_PROGRESS / COMPLETE

# Poll manually:
import time
while status != BatchStatus.COMPLETE:
    time.sleep(30)
    status = processor.get_status(job.id)

results = processor.fetch_results(job.id)
```

## Installation

```bash
pip install ractogateway[openai]     # for OpenAIBatchProcessor
pip install ractogateway[anthropic]  # for AnthropicBatchProcessor
```
