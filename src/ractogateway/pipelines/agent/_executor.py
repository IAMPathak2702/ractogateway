"""Tool registration and execution engine for AgentPipeline.

ToolExecutor wraps a dict of callables and provides:
  - Synchronous execution with timing
  - Async execution (runs sync callables in a thread pool)
  - Parallel async execution via asyncio.gather
  - Human-readable tool descriptions for the system prompt

Built-in tool factories:
  make_finish_tool()        - Always registered; signals task completion
  make_rag_tool(rag)        - Auto-registered when rag_pipeline is provided
  make_sql_tool(sql)        - Auto-registered when sql_pipeline is provided
  make_http_tool()          - Opt-in; fetches URLs via httpx
  make_memory_tools(mem)    - Auto-registered when agent_memory is provided
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from typing import Any

# The tool name that signals the agent is done
FINISH_TOOL = "finish"

# Observations longer than this are truncated before sending back to the LLM
_MAX_OBS_CHARS = 4_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_param_summary(fn: Callable[..., Any]) -> str:
    """Return a compact signature string for the system prompt."""
    try:
        sig = inspect.signature(fn)
        parts: list[str] = []
        for name, param in sig.parameters.items():
            ann = getattr(param.annotation, "__name__", str(param.annotation))
            if ann == "_empty":
                ann = "Any"
            if param.default != inspect.Parameter.empty:
                parts.append(f"{name}: {ann} = {param.default!r}")
            else:
                parts.append(f"{name}: {ann}")
        return ", ".join(parts)
    except (ValueError, TypeError):
        return "..."


def _truncate(text: str, max_chars: int = _MAX_OBS_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n... [{omitted} chars truncated]"


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Runs registered tools by name with sync and async support.

    Parameters
    ----------
    tools:
        Mapping of tool name to callable.
    """

    def __init__(self, tools: dict[str, Callable]) -> None:  # type: ignore[type-arg]
        self._tools = tools

    @property
    def names(self) -> list[str]:
        """Sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def describe_all(self) -> str:
        """Build the tools section for the agent system prompt."""
        lines: list[str] = []
        for name in sorted(self._tools.keys()):
            fn = self._tools[name]
            params = _get_param_summary(fn)
            first_doc_line = (fn.__doc__ or "No description.").strip().splitlines()[0]
            lines.append(f"  {name}({params})\n    -> {first_doc_line}")
        return "\n".join(lines)

    # ── Sync execution ───────────────────────────────────────────────────────

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[str, float]:
        """Execute *tool_name* synchronously.

        Returns
        -------
        tuple[str, float]
            ``(observation, duration_ms)``
        """
        if tool_name not in self._tools:
            return (
                f"ERROR: Unknown tool '{tool_name}'. "
                f"Available: {', '.join(self.names)}",
                0.0,
            )
        t0 = time.perf_counter()
        try:
            result = self._tools[tool_name](**tool_input)
            obs = _truncate(str(result) if result is not None else "OK (no output)")
        except Exception as exc:
            obs = f"ERROR executing '{tool_name}': {type(exc).__name__}: {exc}"
        duration = (time.perf_counter() - t0) * 1000.0
        return obs, duration

    # ── Async execution ───────────────────────────────────────────────────────

    async def aexecute(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[str, float]:
        """Execute *tool_name* asynchronously.

        Async callables are awaited directly; sync callables run in the
        default thread-pool executor to avoid blocking the event loop.

        Returns
        -------
        tuple[str, float]
            ``(observation, duration_ms)``
        """
        if tool_name not in self._tools:
            return (
                f"ERROR: Unknown tool '{tool_name}'. "
                f"Available: {', '.join(self.names)}",
                0.0,
            )
        fn = self._tools[tool_name]
        t0 = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**tool_input)
            else:
                loop = asyncio.get_event_loop()
                captured = dict(tool_input)  # closure-safe copy
                result = await loop.run_in_executor(None, lambda: fn(**captured))
            obs = _truncate(str(result) if result is not None else "OK (no output)")
        except Exception as exc:
            obs = f"ERROR executing '{tool_name}': {type(exc).__name__}: {exc}"
        duration = (time.perf_counter() - t0) * 1000.0
        return obs, duration

    async def aexecute_parallel(
        self,
        calls: list[tuple[str, dict[str, Any]]],
    ) -> list[tuple[str, float]]:
        """Execute multiple tool calls concurrently via ``asyncio.gather``.

        Parameters
        ----------
        calls:
            List of ``(tool_name, tool_input)`` pairs to run in parallel.

        Returns
        -------
        list[tuple[str, float]]
            Results in the same order as *calls*.
        """
        tasks = [self.aexecute(name, inp) for name, inp in calls]
        return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Built-in tool factories
# ---------------------------------------------------------------------------


def make_finish_tool() -> tuple[str, Callable]:  # type: ignore[type-arg]
    """Return the always-present ``finish`` tool.

    When the LLM calls ``finish(answer=...)``, the agent loop stops and
    returns the answer as :attr:`AgentResult.final_answer`.
    """

    def finish(answer: str) -> str:
        """Signal task completion and provide the final answer."""
        return answer

    return FINISH_TOOL, finish


def make_rag_tool(rag_pipeline: Any) -> tuple[str, Callable]:  # type: ignore[type-arg]
    """Return a ``rag_search`` tool backed by a ``RactoRAG`` pipeline."""

    def rag_search(query: str) -> str:
        """Search the knowledge base for information relevant to the query."""
        result = rag_pipeline.search(query)
        # Support RetrievalResult.chunks or plain list
        chunks = (
            getattr(result, "chunks", None) or getattr(result, "results", None) or []
        )
        if chunks:
            parts = []
            for i, chunk in enumerate(chunks[:5], 1):
                text = (
                    getattr(chunk, "text", None)
                    or getattr(chunk, "content", None)
                    or str(chunk)
                )
                parts.append(f"[Result {i}] {text}")
            return "\n\n".join(parts)
        return str(result)

    return "rag_search", rag_search


def make_rag_tool_async(rag_pipeline: Any) -> tuple[str, Callable]:  # type: ignore[type-arg]
    """Return an async ``rag_search`` tool backed by an async ``RactoRAG``."""

    async def rag_search(query: str) -> str:
        """Search the knowledge base for information relevant to the query."""
        result = await rag_pipeline.asearch(query)
        chunks = (
            getattr(result, "chunks", None) or getattr(result, "results", None) or []
        )
        if chunks:
            parts = []
            for i, chunk in enumerate(chunks[:5], 1):
                text = (
                    getattr(chunk, "text", None)
                    or getattr(chunk, "content", None)
                    or str(chunk)
                )
                parts.append(f"[Result {i}] {text}")
            return "\n\n".join(parts)
        return str(result)

    return "rag_search", rag_search


def make_sql_tool(sql_pipeline: Any) -> tuple[str, Callable]:  # type: ignore[type-arg]
    """Return a ``sql_query`` tool backed by a ``SQLAnalystPipeline``."""

    def sql_query(question: str) -> str:
        """Query the connected database using natural language."""
        result = sql_pipeline.run(question)
        if getattr(result, "error", None):
            return f"SQL error: {result.error}"
        return str(getattr(result, "answer", result) or "(no answer)")

    return "sql_query", sql_query


def make_http_tool() -> tuple[str, Callable]:  # type: ignore[type-arg]
    """Return an ``http_get`` tool that fetches URL content via httpx.

    Requires ``httpx``:  ``pip install ractogateway[pipelines-agent-http]``
    """

    def http_get(url: str) -> str:
        """Fetch the text content of a URL (returns up to 4000 chars)."""
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for http_get tool. "
                "Install with: pip install ractogateway[pipelines-agent-http]"
            ) from exc
        try:
            resp = httpx.get(url, timeout=15, follow_redirects=True)
            resp.raise_for_status()
            return _truncate(resp.text)
        except Exception as exc:
            return f"HTTP error: {exc}"

    return "http_get", http_get


def make_memory_tools(
    agent_memory: Any,
) -> list[tuple[str, Callable]]:  # type: ignore[type-arg]
    """Return ``memory_read`` and ``memory_write`` tools backed by *agent_memory*.

    *agent_memory* can be any object supporting::

        memory.get(key) -> Any
        memory.set(key, value) -> None

    or a plain ``dict``.
    """

    def memory_read(key: str) -> str:
        """Read a previously stored value from agent memory by key."""
        val = agent_memory.get(key) if hasattr(agent_memory, "get") else None
        return str(val) if val is not None else "(not found)"

    def memory_write(key: str, value: str) -> str:
        """Store a value in agent memory under a key for later retrieval."""
        if hasattr(agent_memory, "set"):
            agent_memory.set(key, value)
        elif hasattr(agent_memory, "__setitem__"):
            agent_memory[key] = value
        preview_len = 80
        suffix = "..." if len(value) > preview_len else ""
        return f"Stored '{key}' = '{value[:preview_len]}{suffix}'."

    return [("memory_read", memory_read), ("memory_write", memory_write)]
