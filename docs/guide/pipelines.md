# Prebuilt Pipelines

RactoGateway prebuilt pipelines package complete, production-ready workflows
on top of developer kits. A pipeline usually combines multiple LLM calls,
validation, optional retries, and operational controls into one class with
`run()` and `arun()`.

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

# Mail intelligence foundation
pip install "ractogateway[mail]"

# Video Processor - frame extraction, dedup, vision analysis, transcription
pip install "ractogateway[pipelines-video]"
pip install "ractogateway[pipelines-video-whisper]"   # + local faster-whisper
pip install "ractogateway[pipelines-video-yt]"        # + YouTube download
pip install "ractogateway[pipelines-video-full]"      # everything

# Agent Pipeline - no extra deps for core
pip install "ractogateway[pipelines-agent]"
pip install "ractogateway[pipelines-agent-http]"      # + http_get tool
```

## Pipeline Catalog

| Pipeline | Classes | Main job | Best for |
| --- | --- | --- | --- |
| SQL Analyst | `SQLAnalystPipeline`, `AsyncSQLAnalystPipeline` | NL to SQL to analysis to Markdown answer and optional chart | Analytics copilots, BI assistants, ops reporting |
| List Classifier | `ListClassifierPipeline`, `AsyncListClassifierPipeline` | NL query to best matching option(s) from `list[str]` | Ticket routing, intent detection, queue triage |
| Video Processor | `VideoProcessorPipeline`, `AsyncVideoProcessorPipeline` | Video to frames, transcript, vision analysis, summary, and optional RAG | Lecture indexing, whiteboard extraction, video Q&A |
| Mail Intelligence | `RactoMailKit` | Email sync, grounded search, references, saved searches, and DAG orchestration | Mailbox intelligence, vendor follow-up, audit search |
| Agent | `AgentPipeline`, `AsyncAgentPipeline` | ReAct loop: reason, call tool, observe, repeat until done | Multi-step automation, research, data retrieval, agentic workflows |

## Common Import Pattern

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.mail import MockMailConnector, RactoMailKit
from ractogateway.pipelines import (
    AgentPipeline,
    ListClassifierPipeline,
    SQLAnalystPipeline,
    TranscriberBackend,
    VideoProcessorPipeline,
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

mail = RactoMailKit(connectors=[MockMailConnector(messages=[])])
mail.sync()
print(mail.supported_versions())

def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"Sunny, 22 C in {city}"

agent = AgentPipeline(
    kit=gpt.Chat(model="gpt-4o-mini"),
    tools=[get_weather],
    max_steps=6,
)
result = agent.run("What is the weather in Berlin?")
print(result.final_answer)
```

## Detailed Guides

```{toctree}
:maxdepth: 1

pipelines/sql_analyst
pipelines/list_classifier
pipelines/video_processor
mail
pipelines/agent
```

