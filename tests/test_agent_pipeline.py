"""Tests for AgentPipeline — no real API keys required.

All LLM calls are monkey-patched with a fake ``FakeKit`` that returns
pre-programmed JSON responses, letting us exercise every code path without
network access.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    make_sql_tool,
)
from ractogateway.pipelines.agent._executor import _truncate, _get_param_summary
from ractogateway.pipelines.agent.pipeline import _parse_response, _build_transcript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str, input_tokens: int = 10, output_tokens: int = 5) -> Any:
    """Build a fake LLM response object."""
    resp = MagicMock()
    resp.text = text
    resp.content = text
    resp.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return resp


def _json_action(thought: str, tool: str, **kwargs: Any) -> str:
    return json.dumps({"thought": thought, "tool_name": tool, "tool_input": kwargs})


class FakeKit:
    """Synchronous fake kit that returns pre-programmed responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def chat(self, prompt: Any) -> Any:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return _make_response(resp)


class AsyncFakeKit:
    """Asynchronous fake kit that returns pre-programmed responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def achat(self, prompt: Any) -> Any:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return _make_response(resp)


# ---------------------------------------------------------------------------
# _executor helpers
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_string_unchanged(self) -> None:
        assert _truncate("hello", 100) == "hello"

    def test_long_string_truncated(self) -> None:
        result = _truncate("x" * 5000, 4000)
        assert len(result) < 5100
        assert "truncated" in result

    def test_exact_limit_unchanged(self) -> None:
        s = "a" * 4000
        assert _truncate(s, 4000) == s


class TestGetParamSummary:
    def test_simple_function(self) -> None:
        def my_fn(city: str, count: int = 3) -> str:
            return city

        summary = _get_param_summary(my_fn)
        assert "city" in summary
        assert "count" in summary
        assert "3" in summary

    def test_no_params(self) -> None:
        def empty_fn() -> None:
            pass

        assert _get_param_summary(empty_fn) == ""

    def test_unannotated_params(self) -> None:
        def fn(x, y):  # type: ignore[no-untyped-def]
            pass

        summary = _get_param_summary(fn)
        assert "x" in summary
        assert "y" in summary


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------


class TestToolExecutor:
    def _make_executor(self) -> ToolExecutor:
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        def fail_tool(msg: str) -> str:
            """Always raises."""
            raise ValueError(msg)

        _fname, ffn = make_finish_tool()
        return ToolExecutor({"finish": ffn, "add": add, "fail": fail_tool})

    def test_names_sorted(self) -> None:
        ex = self._make_executor()
        assert ex.names == sorted(ex.names)

    def test_describe_all_contains_tool_names(self) -> None:
        ex = self._make_executor()
        desc = ex.describe_all()
        assert "add" in desc
        assert "fail" in desc
        assert "finish" in desc

    def test_execute_known_tool(self) -> None:
        ex = self._make_executor()
        obs, ms = ex.execute("add", {"a": 3, "b": 4})
        assert obs == "7"
        assert ms >= 0

    def test_execute_unknown_tool(self) -> None:
        ex = self._make_executor()
        obs, ms = ex.execute("nonexistent", {})
        assert "Unknown tool" in obs
        assert ms == 0.0

    def test_execute_tool_raises(self) -> None:
        ex = self._make_executor()
        obs, _ = ex.execute("fail", {"msg": "boom"})
        assert "ERROR" in obs
        assert "boom" in obs

    def test_aexecute_sync_tool(self) -> None:
        ex = self._make_executor()

        async def _run() -> tuple[str, float]:
            return await ex.aexecute("add", {"a": 1, "b": 2})

        obs, _ms = asyncio.get_event_loop().run_until_complete(_run())
        assert obs == "3"

    def test_aexecute_unknown_tool(self) -> None:
        ex = self._make_executor()

        async def _run() -> tuple[str, float]:
            return await ex.aexecute("nope", {})

        obs, ms = asyncio.get_event_loop().run_until_complete(_run())
        assert "Unknown tool" in obs
        assert ms == 0.0

    def test_aexecute_parallel(self) -> None:
        ex = self._make_executor()

        async def _run() -> list[tuple[str, float]]:
            return await ex.aexecute_parallel([("add", {"a": 1, "b": 1}), ("add", {"a": 2, "b": 2})])

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert len(results) == 2
        assert results[0][0] == "2"
        assert results[1][0] == "4"

    def test_aexecute_async_coroutine_tool(self) -> None:
        async def async_greet(name: str) -> str:
            """Greet async."""
            return f"Hello {name}"

        ex = ToolExecutor({"greet": async_greet})

        async def _run() -> tuple[str, float]:
            return await ex.aexecute("greet", {"name": "World"})

        obs, _ = asyncio.get_event_loop().run_until_complete(_run())
        assert "Hello World" in obs


# ---------------------------------------------------------------------------
# Built-in tool factories
# ---------------------------------------------------------------------------


class TestMakeFinishTool:
    def test_returns_answer(self) -> None:
        name, fn = make_finish_tool()
        assert name == FINISH_TOOL
        assert fn(answer="done") == "done"


class TestMakeRagTool:
    def test_formats_chunks(self) -> None:
        chunk = MagicMock()
        chunk.text = "The sky is blue."
        result_mock = MagicMock()
        result_mock.chunks = [chunk]
        rag = MagicMock()
        rag.search.return_value = result_mock

        name, fn = make_rag_tool(rag)
        assert name == "rag_search"
        out = fn(query="sky color")
        assert "sky is blue" in out
        rag.search.assert_called_once_with("sky color")

    def test_fallback_to_str(self) -> None:
        result_mock = MagicMock()
        result_mock.chunks = []
        result_mock.results = []
        rag = MagicMock()
        rag.search.return_value = result_mock

        _, fn = make_rag_tool(rag)
        # Should not raise; falls back to str(result)
        out = fn(query="anything")
        assert isinstance(out, str)


class TestMakeSqlTool:
    def test_returns_answer(self) -> None:
        sql_result = MagicMock()
        sql_result.error = None
        sql_result.answer = "42 rows"
        pipeline = MagicMock()
        pipeline.run.return_value = sql_result

        name, fn = make_sql_tool(pipeline)
        assert name == "sql_query"
        out = fn(question="How many rows?")
        assert "42 rows" in out

    def test_returns_error(self) -> None:
        sql_result = MagicMock()
        sql_result.error = "syntax error"
        pipeline = MagicMock()
        pipeline.run.return_value = sql_result

        _, fn = make_sql_tool(pipeline)
        out = fn(question="bad query")
        assert "SQL error" in out


class TestMakeHttpTool:
    def test_http_get_success(self) -> None:
        name, fn = make_http_tool()
        assert name == "http_get"

        mock_resp = MagicMock()
        mock_resp.text = "Hello from the web!"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            out = fn(url="https://example.com")
        assert "Hello from the web" in out

    def test_http_get_error(self) -> None:
        _, fn = make_http_tool()
        with patch("httpx.get", side_effect=RuntimeError("timeout")):
            out = fn(url="https://bad.example")
        assert "HTTP error" in out

    def test_http_missing_httpx(self) -> None:
        import builtins
        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "httpx":
                raise ImportError("no httpx")
            return real_import(name, *args, **kwargs)

        _, fn = make_http_tool()
        with patch("builtins.__import__", side_effect=mock_import), pytest.raises(ImportError, match="httpx"):
            fn(url="https://example.com")


class TestMakeMemoryTools:
    def test_set_and_get_dict(self) -> None:
        mem: dict[str, str] = {}
        tools = dict(make_memory_tools(mem))

        assert "memory_write" in tools
        assert "memory_read" in tools

        out_write = tools["memory_write"](key="city", value="Paris")
        assert "Paris" in out_write

        out_read = tools["memory_read"](key="city")
        assert "Paris" in out_read

    def test_read_missing_key(self) -> None:
        mem: dict[str, str] = {}
        tools = dict(make_memory_tools(mem))
        out = tools["memory_read"](key="missing")
        assert "not found" in out

    def test_set_with_object_api(self) -> None:
        class Store:
            def __init__(self) -> None:
                self._d: dict[str, str] = {}

            def get(self, key: str) -> str | None:
                return self._d.get(key)

            def set(self, key: str, value: str) -> None:
                self._d[key] = value

        store = Store()
        tools = dict(make_memory_tools(store))
        tools["memory_write"](key="x", value="42")
        assert tools["memory_read"](key="x") == "42"


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_pure_json(self) -> None:
        text = '{"thought": "ok", "tool_name": "add", "tool_input": {"a": 1, "b": 2}}'
        thought, calls = _parse_response(text)
        assert thought == "ok"
        assert len(calls) == 1
        tool, inp = calls[0]
        assert tool == "add"
        assert inp == {"a": 1, "b": 2}

    def test_markdown_fence(self) -> None:
        text = '```json\n{"thought": "t", "tool_name": "finish", "tool_input": {"answer": "done"}}\n```'
        _thought, calls = _parse_response(text)
        tool, inp = calls[0]
        assert tool == "finish"
        assert inp["answer"] == "done"

    def test_embedded_json(self) -> None:
        text = 'Here is my action:\n{"thought": "x", "tool_name": "foo", "tool_input": {}}\nDone.'
        _, calls = _parse_response(text)
        tool, _inp = calls[0]
        assert tool == "foo"

    def test_malformed_fallback(self) -> None:
        text = "This is not JSON at all"
        _thought, calls = _parse_response(text)
        tool, inp = calls[0]
        assert tool == FINISH_TOOL
        assert text in inp.get("answer", "")

    def test_non_dict_tool_input_coerced(self) -> None:
        text = '{"thought": "t", "tool_name": "foo", "tool_input": "just a string"}'
        _, calls = _parse_response(text)
        _tool, inp = calls[0]
        assert isinstance(inp, dict)

    def test_missing_thought(self) -> None:
        text = '{"tool_name": "finish", "tool_input": {"answer": "x"}}'
        thought, calls = _parse_response(text)
        tool, _inp = calls[0]
        assert thought is None
        assert tool == "finish"


# ---------------------------------------------------------------------------
# _build_transcript
# ---------------------------------------------------------------------------


class TestBuildTranscript:
    def test_no_steps(self) -> None:
        t = _build_transcript("Find capital of France", [])
        assert "Find capital of France" in t
        assert "Your next action" in t

    def test_with_steps(self) -> None:
        step = AgentStep(
            step_num=1,
            thought="Searching...",
            tool_name="search",
            tool_input={"q": "France capital"},
            observation="Paris",
        )
        t = _build_transcript("goal", [step])
        assert "search" in t
        assert "Paris" in t
        assert "Your next action" in t


# ---------------------------------------------------------------------------
# AgentResult helpers
# ---------------------------------------------------------------------------


class TestAgentResult:
    def _make_result(self) -> AgentResult:
        steps = [
            AgentStep(step_num=1, tool_name="search", tool_input={"q": "test"}, observation="result"),
            AgentStep(step_num=2, tool_name=FINISH_TOOL, tool_input={"answer": "42"}, is_finish=True),
        ]
        usage = AgentUsage(total_input_tokens=100, total_output_tokens=50, steps_taken=2, tools_called=1)
        return AgentResult(
            goal="Find the answer",
            final_answer="42",
            steps=steps,
            stop_reason=StopReason.FINISHED,
            usage=usage,
        )

    def test_succeeded(self) -> None:
        assert self._make_result().succeeded() is True

    def test_total_tokens(self) -> None:
        r = self._make_result()
        assert r.usage.total_tokens == 150

    def test_get_tool_calls_excludes_finish(self) -> None:
        r = self._make_result()
        calls = r.get_tool_calls()
        assert len(calls) == 1
        assert calls[0][0] == "search"

    def test_get_observations(self) -> None:
        r = self._make_result()
        obs = r.get_observations()
        assert obs == ["result"]

    def test_to_json_string(self) -> None:
        r = self._make_result()
        txt = r.to_json()
        assert txt is not None
        data = json.loads(txt)
        assert data["goal"] == "Find the answer"

    def test_to_json_file(self, tmp_path: Any) -> None:
        r = self._make_result()
        path = str(tmp_path / "result.json")
        ret = r.to_json(path)
        assert ret is None
        import pathlib
        content = pathlib.Path(path).read_text(encoding="utf-8")
        assert "Find the answer" in content

    def test_to_markdown_contains_sections(self) -> None:
        r = self._make_result()
        md = r.to_markdown()
        assert md is not None
        assert "# Agent Run" in md
        assert "Final Answer" in md
        assert "42" in md

    def test_to_markdown_file(self, tmp_path: Any) -> None:
        r = self._make_result()
        path = str(tmp_path / "result.md")
        ret = r.to_markdown(path)
        assert ret is None

    def test_not_succeeded_max_steps(self) -> None:
        r = AgentResult(goal="g", stop_reason=StopReason.MAX_STEPS)
        assert r.succeeded() is False

    def test_not_succeeded_error(self) -> None:
        r = AgentResult(goal="g", stop_reason=StopReason.ERROR, error="oops")
        assert r.succeeded() is False


# ---------------------------------------------------------------------------
# AgentPipeline — sync run()
# ---------------------------------------------------------------------------


class TestAgentPipelineSync:
    def _make_agent(self, responses: list[str], **kwargs: Any) -> AgentPipeline:
        kit = FakeKit(responses)
        return AgentPipeline(kit=kit, **kwargs)

    def test_single_step_finish(self) -> None:
        resp = _json_action("Done", "finish", answer="Paris")
        agent = self._make_agent([resp])
        result = agent.run("Capital of France?")
        assert result.succeeded()
        assert result.final_answer == "Paris"
        assert result.stop_reason == StopReason.FINISHED
        assert len(result.steps) == 1

    def test_tool_then_finish(self) -> None:
        def get_city(name: str) -> str:
            """Return city info."""
            return f"{name} is the capital of France."

        step1 = _json_action("Need info", "get_city", name="Paris")
        step2 = _json_action("Got it", "finish", answer="Paris is the capital.")
        agent = self._make_agent([step1, step2], tools=[get_city])

        result = agent.run("What is the capital of France?")
        assert result.succeeded()
        assert result.usage.tools_called == 1
        assert result.usage.steps_taken == 2

    def test_max_steps_reached(self) -> None:
        # Keeps calling a known tool forever — never calls finish.
        # Use a registered tool so no errors occur (avoids circuit breaker).
        def noop(x: str = "") -> str:
            """No-op tool."""
            return "ok"

        resp = _json_action("Thinking...", "noop")
        agent = self._make_agent([resp], max_steps=3, tools=[noop])
        result = agent.run("Impossible task")
        assert result.stop_reason == StopReason.MAX_STEPS
        assert result.final_answer is None
        assert result.usage.steps_taken == 3

    def test_unknown_tool_observation(self) -> None:
        resp = _json_action("Using unknown", "unknown_tool")
        finish = _json_action("Observed error", "finish", answer="Error handled")
        agent = self._make_agent([resp, finish])
        result = agent.run("Any task")
        # Step 1 observation should contain ERROR
        assert "Unknown tool" in result.steps[0].observation

    def test_safe_mode_catches_exception(self) -> None:
        kit = MagicMock()
        kit.chat.side_effect = RuntimeError("LLM exploded")
        agent = AgentPipeline(kit=kit, safe_mode=True)
        result = agent.run("Test safe mode")
        assert result.stop_reason == StopReason.ERROR
        assert "LLM exploded" in (result.error or "")

    def test_safe_mode_false_raises(self) -> None:
        kit = MagicMock()
        kit.chat.side_effect = RuntimeError("explode")
        agent = AgentPipeline(kit=kit, safe_mode=False)
        with pytest.raises(RuntimeError, match="explode"):
            agent.run("Task")

    def test_per_call_max_steps_override(self) -> None:
        resp = _json_action("Looping", "noop")
        agent = self._make_agent([resp], max_steps=10)
        result = agent.run("Loop task", max_steps=2)
        assert result.stop_reason == StopReason.MAX_STEPS
        assert result.usage.steps_taken == 2

    def test_registered_tools_contains_finish(self) -> None:
        agent = self._make_agent(["{}"])
        assert FINISH_TOOL in agent.registered_tools

    def test_system_prompt_property(self) -> None:
        agent = self._make_agent(["{}"])
        assert "AVAILABLE TOOLS" in agent.system_prompt

    def test_custom_system_prompt(self) -> None:
        agent = self._make_agent(["{}"], system_prompt="Custom prompt here")
        assert agent.system_prompt == "Custom prompt here"

    def test_extra_rules_appended(self) -> None:
        agent = self._make_agent(["{}"], extra_rules="Never answer in French.")
        assert "Never answer in French" in agent.system_prompt

    def test_token_usage_accumulated(self) -> None:
        step1 = _json_action("Think", "noop")
        step2 = _json_action("Done", "finish", answer="ok")
        kit = FakeKit([step1, step2])
        agent = AgentPipeline(kit=kit)
        result = agent.run("Task")
        # Both steps had input_tokens=10, output_tokens=5
        assert result.usage.total_input_tokens == 20
        assert result.usage.total_output_tokens == 10

    def test_malformed_llm_response_fallback(self) -> None:
        agent = self._make_agent(["not json at all"])
        result = agent.run("Task")
        assert result.succeeded()  # Fallback: treats raw text as finish answer

    def test_rate_limiter_blocked(self) -> None:
        limiter = MagicMock()
        limiter.check_and_consume.return_value = False
        limiter.get_remaining.return_value = 0

        agent = AgentPipeline(
            kit=FakeKit([]),
            rate_limiter=limiter,
        )
        with pytest.raises(AgentRateLimitExceededError):
            agent.run("Task")

    def test_rate_limiter_per_call_user_id(self) -> None:
        limiter = MagicMock()
        limiter.check_and_consume.return_value = False
        limiter.get_remaining.return_value = 0

        agent = AgentPipeline(kit=FakeKit([]), rate_limiter=limiter)
        with pytest.raises(AgentRateLimitExceededError):
            agent.run("Task", user_id="alice")
        limiter.check_and_consume.assert_called_with("alice", 1)

    def test_rag_pipeline_auto_registers_tool(self) -> None:
        chunk = MagicMock()
        chunk.text = "Found it."
        rag_result = MagicMock()
        rag_result.chunks = [chunk]
        rag = MagicMock()
        rag.search.return_value = rag_result

        step1 = _json_action("Search RAG", "rag_search", query="test")
        step2 = _json_action("Done", "finish", answer="Found it.")
        agent = AgentPipeline(kit=FakeKit([step1, step2]), rag_pipeline=rag)

        assert "rag_search" in agent.registered_tools
        result = agent.run("Search for test")
        assert result.succeeded()

    def test_sql_pipeline_auto_registers_tool(self) -> None:
        sql_result = MagicMock()
        sql_result.error = None
        sql_result.answer = "100 rows"
        sql = MagicMock()
        sql.run.return_value = sql_result

        step1 = _json_action("Query DB", "sql_query", question="count rows")
        step2 = _json_action("Done", "finish", answer="100 rows")
        agent = AgentPipeline(kit=FakeKit([step1, step2]), sql_pipeline=sql)

        assert "sql_query" in agent.registered_tools
        result = agent.run("How many rows?")
        assert result.succeeded()

    def test_memory_tools_auto_registered(self) -> None:
        mem: dict[str, str] = {}
        agent = AgentPipeline(kit=FakeKit([]), agent_memory=mem)
        assert "memory_read" in agent.registered_tools
        assert "memory_write" in agent.registered_tools

    def test_user_tool_shadows_builtin(self) -> None:
        """A user tool named 'rag_search' should override the rag tool."""
        def rag_search(query: str) -> str:  # type: ignore[return]
            """Custom RAG."""
            return "custom result"

        rag = MagicMock()
        agent = AgentPipeline(
            kit=FakeKit([]),
            rag_pipeline=rag,
            tools=[rag_search],
        )
        # User tool registered last, so it should be the active one
        assert "rag_search" in agent.registered_tools

    def test_user_tool_with_tool_name_attr(self) -> None:
        def my_fn(x: str) -> str:
            """Custom."""
            return x

        my_fn.__tool_name__ = "custom_name"  # type: ignore[attr-defined]
        agent = AgentPipeline(kit=FakeKit([]), tools=[my_fn])
        assert "custom_name" in agent.registered_tools


# ---------------------------------------------------------------------------
# AgentPipeline — async arun()
# ---------------------------------------------------------------------------


class TestAgentPipelineAsync:
    def _run(self, coro: Any) -> Any:
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_single_step_finish_async(self) -> None:
        resp = _json_action("Done", "finish", answer="Tokyo")
        kit = AsyncFakeKit([resp])
        agent = AgentPipeline(kit=kit)
        result = self._run(agent.arun("Capital of Japan?"))
        assert result.succeeded()
        assert result.final_answer == "Tokyo"

    def test_tool_then_finish_async(self) -> None:
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        step1 = _json_action("Calculate", "multiply", a=6, b=7)
        step2 = _json_action("Done", "finish", answer="42")
        kit = AsyncFakeKit([step1, step2])
        agent = AgentPipeline(kit=kit, tools=[multiply])
        result = self._run(agent.arun("6 times 7?"))
        assert result.succeeded()
        assert result.usage.tools_called == 1

    def test_max_steps_async(self) -> None:
        def noop(x: str = "") -> str:
            """No-op tool."""
            return "ok"

        resp = _json_action("Looping", "noop")
        kit = AsyncFakeKit([resp])
        agent = AgentPipeline(kit=kit, max_steps=2, tools=[noop])
        result = self._run(agent.arun("Loop"))
        assert result.stop_reason == StopReason.MAX_STEPS

    def test_safe_mode_async(self) -> None:
        kit = MagicMock()
        kit.achat = AsyncMock(side_effect=RuntimeError("async explode"))
        agent = AgentPipeline(kit=kit, safe_mode=True)
        result = self._run(agent.arun("Task"))
        assert result.stop_reason == StopReason.ERROR
        assert "async explode" in (result.error or "")

    def test_async_tool_awaited(self) -> None:
        async def async_tool(x: str) -> str:
            """Async tool."""
            return f"async:{x}"

        step1 = _json_action("Use async", "async_tool", x="hello")
        step2 = _json_action("Done", "finish", answer="async:hello")
        kit = AsyncFakeKit([step1, step2])
        agent = AgentPipeline(kit=kit, tools=[async_tool])
        result = self._run(agent.arun("Test async tool"))
        assert result.succeeded()
        assert "async:hello" in result.steps[0].observation


# ---------------------------------------------------------------------------
# AsyncAgentPipeline
# ---------------------------------------------------------------------------


class TestAsyncAgentPipeline:
    def _run(self, coro: Any) -> Any:
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_run_delegates_to_inner(self) -> None:
        resp = _json_action("Done", "finish", answer="42")
        kit = AsyncFakeKit([resp])
        agent = AsyncAgentPipeline(kit=kit)
        result = self._run(agent.run("What is 6 * 7?"))
        assert result.succeeded()

    def test_registered_tools_proxy(self) -> None:
        kit = AsyncFakeKit([])
        agent = AsyncAgentPipeline(kit=kit)
        assert FINISH_TOOL in agent.registered_tools

    def test_system_prompt_proxy(self) -> None:
        kit = AsyncFakeKit([])
        agent = AsyncAgentPipeline(kit=kit)
        assert "AVAILABLE TOOLS" in agent.system_prompt


# ---------------------------------------------------------------------------
# Import from pipelines top-level
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    def test_import_from_pipelines(self) -> None:
        from ractogateway.pipelines import AgentPipeline as PipelineAgent
        from ractogateway.pipelines import AsyncAgentPipeline as AsyncPipelineAgent
        from ractogateway.pipelines import AgentResult as PipelineAgentResult
        from ractogateway.pipelines import StopReason as PipelineStopReason

        assert PipelineAgent is AgentPipeline
        assert AsyncPipelineAgent is AsyncAgentPipeline
        assert PipelineAgentResult is AgentResult
        assert PipelineStopReason is StopReason

    def test_import_tool_factories(self) -> None:
        from ractogateway.pipelines import make_finish_tool as pipelines_make_finish_tool
        from ractogateway.pipelines import ToolExecutor as PipelineToolExecutor

        assert pipelines_make_finish_tool is make_finish_tool
        assert PipelineToolExecutor is ToolExecutor
