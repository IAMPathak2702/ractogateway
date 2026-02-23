from __future__ import annotations

import pytest

from ractogateway._tool_runtime import (
    build_tool_followup_user_message,
    execute_tool_calls_async,
    execute_tool_calls_sync,
)
from ractogateway.adapters.base import ToolCallResult
from ractogateway.tools.registry import ToolRegistry


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @registry.register
    async def greet(name: str) -> str:
        """Greet a user."""
        return f"Hello {name}"

    return registry


def test_execute_tool_calls_sync_handles_registered_and_missing_tools() -> None:
    registry = _make_registry()
    calls = [
        ToolCallResult(name="add", arguments={"a": 12, "b": 8}),
        ToolCallResult(name="missing", arguments={}),
    ]

    results = execute_tool_calls_sync(calls, registry)

    assert results[0] == "20"
    assert "no callable registered" in results[1]


def test_execute_tool_calls_sync_reports_async_callable_in_sync_mode() -> None:
    registry = _make_registry()
    calls = [ToolCallResult(name="greet", arguments={"name": "Ada"})]

    results = execute_tool_calls_sync(calls, registry)

    assert "returned awaitable in sync mode" in results[0]


@pytest.mark.asyncio
async def test_execute_tool_calls_async_supports_sync_and_async_callables() -> None:
    registry = _make_registry()
    calls = [
        ToolCallResult(name="add", arguments={"a": 2, "b": 3}),
        ToolCallResult(name="greet", arguments={"name": "Ada"}),
    ]

    results = await execute_tool_calls_async(calls, registry)

    assert results == ["5", "Hello Ada"]


def test_build_tool_followup_user_message_includes_original_and_results() -> None:
    calls = [ToolCallResult(name="add", arguments={"a": 12, "b": 8})]
    message = build_tool_followup_user_message(
        original_user_message="What is 12 * 8?",
        tool_calls=calls,
        results=["96"],
    )

    assert "Original user request: What is 12 * 8?" in message
    assert "add(a=12, b=8) -> 96" in message
    assert "provide the best final answer" in message
