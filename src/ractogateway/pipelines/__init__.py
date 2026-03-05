"""RactoGateway Prebuilt Pipelines.

Prebuilt end-to-end pipelines for real-world LLM workflows.
Each pipeline is a self-contained class with ``run()`` / ``arun()`` methods,
full observability hooks, per-step model control, and optional per-call overrides.

Available pipelines
-------------------
- :class:`SQLAnalystPipeline` / :class:`AsyncSQLAnalystPipeline` —
  NL → SQL → pandas → Markdown answer + optional Plotly chart.
  Requires: ``pip install ractogateway[pipelines-sql]``
  Charts:   ``pip install ractogateway[pipelines-sql-viz]``

- :class:`ListClassifierPipeline` / :class:`AsyncListClassifierPipeline` —
  NL query → best-matching item(s) from a ``list[str]``.
  Uses dynamic ``Enum`` + Pydantic validation; supports single/multi selection,
  confidence scores, reasoning, retries, memory, rate limiting, and telemetry.
  No extra dependencies — works with any installed provider kit.

- :class:`VideoProcessorPipeline` / :class:`AsyncVideoProcessorPipeline` —
  Process tutorial/lecture videos: frame deduplication, audio transcription,
  vision-LLM analysis (whiteboard / screen), comprehensive summary, optional
  RAG storage.  Accepts local paths, URLs, YouTube links, raw bytes, or
  pre-extracted frame images.
  Requires: ``pip install ractogateway[pipelines-video]``
  Audio:    ``pip install ractogateway[pipelines-video-whisper]``
  YouTube:  ``pip install ractogateway[pipelines-video-yt]``

- :class:`AgentPipeline` / :class:`AsyncAgentPipeline` —
  Autonomous ReAct (Reason + Act) agent with pluggable tools.  The agent
  reasons step-by-step, calls tools (RAG search, SQL query, HTTP fetch,
  memory, or any Python callable), observes results, and repeats until it
  reaches a final answer or the ``max_steps`` cap.
  No extra dependencies for the core agent.
  HTTP tool: ``pip install ractogateway[pipelines-agent-http]``

Usage::

    from ractogateway.pipelines import SQLAnalystPipeline, ListClassifierPipeline
    from ractogateway.openai_developer_kit import Chat

    # SQL Analyst
    pipeline = SQLAnalystPipeline(kit=Chat(model="gpt-4o"))
    result = pipeline.run(
        user_query="Top 5 products by quantity sold?",
        connection_string="postgresql://user:pass@localhost/db",
    )
    print(result.answer)

    # List Classifier
    classifier = ListClassifierPipeline(
        kit=Chat(model="gpt-4o-mini"),
        options=["Billing", "Technical Support", "Sales", "Account Management"],
        selection_mode="single",
        include_confidence=True,
        include_reasoning=True,
    )
    result = classifier.run("I can't log into my account")
    print(result.first)           # "Account Management"
    print(result.top_confidence)  # 0.94
    print(result.as_dict())       # {"selected": [...], "confidences": [...], ...}

    # Video Processor
    from ractogateway.pipelines import VideoProcessorPipeline, TranscriberBackend
    vp = VideoProcessorPipeline(
        kit=Chat(model="gpt-4o"),
        fps=1.0,
        similarity_threshold=85.0,
        transcriber=TranscriberBackend.FASTER_WHISPER,
        transcriber_model="base",
        generate_summary=True,
    )
    res = vp.run("lecture.mp4")       # or YouTube URL / bytes / pre-extracted frames
    print(res.summary)
    res.to_markdown("report.md")

    # Agent
    from ractogateway.pipelines import AgentPipeline

    def get_weather(city: str) -> str:
        \"\"\"Return the current weather for a city.\"\"\"
        return f"Sunny, 22 C in {city}"

    agent = AgentPipeline(
        kit=Chat(model="gpt-4o"),
        tools=[get_weather],
        max_steps=6,
        safe_mode=True,
    )
    result = agent.run("What is the weather in Paris?")
    print(result.final_answer)
    print(result.to_markdown())
"""

from ractogateway.pipelines.agent import (
    FINISH_TOOL,
    AgentPipeline,
    AgentRateLimitExceededError,
    AgentResult,
    AgentStep,
    AgentUsage,
    AsyncAgentPipeline,
    StopReason,
    ToolExecutor,
    make_finish_tool,
    make_http_tool,
    make_memory_tools,
    make_rag_tool,
    make_rag_tool_async,
    make_sql_tool,
)
from ractogateway.pipelines.list_classifier import (
    AsyncListClassifierPipeline,
    AuditEntry,
    ClassifierRateLimitExceededError,
    ClassifierResult,
    ClassifierUsage,
    ListClassifierPipeline,
)
from ractogateway.pipelines.sql_analyst import (
    AsyncSQLAnalystPipeline,
    ChartSpec,
    PipelineUsage,
    RateLimitExceededError,
    ReadOnlySQLGuard,
    ReadOnlyViolationError,
    SQLAnalystPipeline,
    SQLAnalystResult,
    clear_schema_cache,
)
from ractogateway.pipelines.video_processor import (
    AsyncVideoProcessorPipeline,
    DeduplicationMethod,
    FrameAnalysisMode,
    FrameEntry,
    TranscriberBackend,
    TranscriptSegment,
    VideoConfig,
    VideoInput,
    VideoProcessorPipeline,
    VideoProcessorResult,
    VideoProcessorUsage,
    VideoRateLimitExceededError,
    VideoSection,
)

__all__ = [
    # Agent
    "AgentPipeline",
    "AgentRateLimitExceededError",
    "AgentResult",
    "AgentStep",
    "AgentUsage",
    "AsyncAgentPipeline",
    "FINISH_TOOL",
    "StopReason",
    "ToolExecutor",
    "make_finish_tool",
    "make_http_tool",
    "make_memory_tools",
    "make_rag_tool",
    "make_rag_tool_async",
    "make_sql_tool",
    # SQL Analyst
    "AsyncSQLAnalystPipeline",
    "ChartSpec",
    "PipelineUsage",
    "RateLimitExceededError",
    "ReadOnlySQLGuard",
    "ReadOnlyViolationError",
    "SQLAnalystPipeline",
    "SQLAnalystResult",
    "clear_schema_cache",
    # List Classifier
    "AsyncListClassifierPipeline",
    "AuditEntry",
    "ClassifierRateLimitExceededError",
    "ClassifierResult",
    "ClassifierUsage",
    "ListClassifierPipeline",
    # Video Processor
    "AsyncVideoProcessorPipeline",
    "DeduplicationMethod",
    "FrameAnalysisMode",
    "FrameEntry",
    "TranscriberBackend",
    "TranscriptSegment",
    "VideoConfig",
    "VideoInput",
    "VideoProcessorPipeline",
    "VideoProcessorResult",
    "VideoProcessorUsage",
    "VideoRateLimitExceededError",
    "VideoSection",
]
