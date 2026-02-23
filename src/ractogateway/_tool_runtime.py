"""Internal helpers for local tool execution loops in chat kits."""

from __future__ import annotations

import inspect

from ractogateway.adapters.base import ToolCallResult
from ractogateway.tools.registry import ToolRegistry


def execute_tool_calls_sync(
    tool_calls: list[ToolCallResult],
    registry: ToolRegistry,
) -> list[str]:
    """Execute tool calls synchronously and return stringified results."""
    results: list[str] = []
    for tc in tool_calls:
        fn = registry.get_callable(tc.name)
        if fn is None:
            results.append(f"[Error: no callable registered for tool {tc.name!r}]")
            continue
        try:
            raw = fn(**tc.arguments)
            if inspect.isawaitable(raw):
                close = getattr(raw, "close", None)
                if callable(close):
                    close()
                results.append(
                    f"[Error executing {tc.name!r}: returned awaitable in sync mode. "
                    "Use achat() or register a sync callable.]"
                )
            else:
                results.append(str(raw))
        except Exception as exc:  # pragma: no cover
            results.append(f"[Error executing {tc.name!r}: {exc}]")
    return results


async def execute_tool_calls_async(
    tool_calls: list[ToolCallResult],
    registry: ToolRegistry,
) -> list[str]:
    """Execute tool calls asynchronously and return stringified results."""
    results: list[str] = []
    for tc in tool_calls:
        fn = registry.get_callable(tc.name)
        if fn is None:
            results.append(f"[Error: no callable registered for tool {tc.name!r}]")
            continue
        try:
            raw = fn(**tc.arguments)
            if inspect.isawaitable(raw):
                raw = await raw
            results.append(str(raw))
        except Exception as exc:  # pragma: no cover
            results.append(f"[Error executing {tc.name!r}: {exc}]")
    return results


def build_tool_followup_user_message(
    *,
    original_user_message: str,
    tool_calls: list[ToolCallResult],
    results: list[str],
) -> str:
    """Build a provider-agnostic follow-up user message from tool outputs."""
    lines: list[str] = [
        "You requested tools in the previous step. The tools have now been executed.",
        f"Original user request: {original_user_message}",
        "Tool execution results:",
    ]
    for tc, result in zip(tool_calls, results, strict=True):
        args_repr = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
        lines.append(f"- {tc.name}({args_repr}) -> {result}")
    lines.append(
        "Now provide the best final answer to the original user request using these results."
    )
    return "\n".join(lines)
