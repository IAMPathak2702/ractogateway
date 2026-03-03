# Prebuilt Pipelines

RactoGateway prebuilt pipelines package complete, production-ready workflows on top
of developer kits. A pipeline usually combines multiple LLM calls, validation,
optional retries, and operational controls into one class with `run()` and `arun()`.

## Installation

```bash
# All pipelines bundle
pip install "ractogateway[pipelines]"

# SQL Analyst only (requires sqlalchemy + pandas)
pip install "ractogateway[pipelines-sql]"

# SQL Analyst + Plotly chart support
pip install "ractogateway[pipelines-sql-viz]"

# SQL Analyst with Polars analysis engine
pip install "ractogateway[pipelines-sql-polars]"

# List Classifier only (no extra deps beyond core package)
pip install "ractogateway[pipelines-classifier]"

# Video Processor — frame extraction, dedup, vision analysis, transcription
pip install "ractogateway[pipelines-video]"
pip install "ractogateway[pipelines-video-whisper]"   # + local faster-whisper
pip install "ractogateway[pipelines-video-yt]"        # + YouTube download
pip install "ractogateway[pipelines-video-full]"      # everything
```

## Pipeline Catalog

| Pipeline | Classes | Main job | Best for |
| --- | --- | --- | --- |
| SQL Analyst | `SQLAnalystPipeline`, `AsyncSQLAnalystPipeline` | Natural language → SQL → analysis → Markdown answer + optional chart | Analytics copilots, BI assistants, ops reporting |
| List Classifier | `ListClassifierPipeline`, `AsyncListClassifierPipeline` | Natural language query → best matching option(s) from `list[str]` | Ticket routing, intent detection, queue triage |
| Video Processor | `VideoProcessorPipeline`, `AsyncVideoProcessorPipeline` | Video → frames + transcript + vision analysis + summary + RAG | Lecture indexing, whiteboard extraction, video Q&A |

## Common Import Pattern

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.pipelines import (
    SQLAnalystPipeline,
    ListClassifierPipeline,
    VideoProcessorPipeline,
    TranscriberBackend,
)

sql_pipeline = SQLAnalystPipeline(kit=gpt.Chat(model="gpt-4o"))

classifier = ListClassifierPipeline(
    kit=gpt.Chat(model="gpt-4o-mini"),
    options=["Billing", "Technical Support", "Sales"],
)

video_pipeline = VideoProcessorPipeline(
    kit=gpt.Chat(model="gpt-4o"),
    transcriber=TranscriberBackend.FASTER_WHISPER,
    generate_summary=True,
)
result = video_pipeline.run("lecture.mp4")
print(result.summary)
```

## Detailed Guides

```{toctree}
:maxdepth: 1

pipelines/sql_analyst
pipelines/list_classifier
pipelines/video_processor
```
