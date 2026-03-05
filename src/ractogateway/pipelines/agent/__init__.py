"""AgentPipeline — autonomous ReAct agent with pluggable tools.

Exports
-------
AgentPipeline
    Sync ``run()`` + async ``arun()`` ReAct loop.
AsyncAgentPipeline
    Async-only wrapper suitable for FastAPI / async servers.
AgentResult, AgentStep, AgentUsage, StopReason
    Result and step models.
AgentRateLimitExceededError
    Raised when the rate limiter blocks a run.

Tool factories (for advanced use)
----------------------------------
make_finish_tool, make_rag_tool, make_rag_tool_async, make_sql_tool, make_http_tool, make_memory_tools
    Pre-built tool factories used internally; exposed for custom wiring.
ToolExecutor
    Low-level sync / async tool runner.

Install
-------
No extra dependencies required for the core agent.
HTTP tool requires ``httpx``::

    pip install ractogateway[pipelines-agent-http]
"""

from ._executor import (
    FINISH_TOOL,
    ToolExecutor,
    make_finish_tool,
    make_http_tool,
    make_memory_tools,
    make_rag_tool,
    make_rag_tool_async,
    make_sql_tool,
)
from ._models import (
    AgentRateLimitExceededError,
    AgentResult,
    AgentStep,
    AgentUsage,
    StopReason,
)
from .pipeline import AgentPipeline, AsyncAgentPipeline

__all__ = [
    # Main classes
    "AgentPipeline",
    "AsyncAgentPipeline",
    # Models
    "AgentResult",
    "AgentStep",
    "AgentUsage",
    "StopReason",
    # Exceptions
    "AgentRateLimitExceededError",
    # Tool factories (advanced)
    "FINISH_TOOL",
    "ToolExecutor",
    "make_finish_tool",
    "make_http_tool",
    "make_memory_tools",
    "make_rag_tool",
    "make_rag_tool_async",
    "make_sql_tool",
]
