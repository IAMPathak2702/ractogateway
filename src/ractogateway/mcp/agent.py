"""MCPAgent — agentic tool-execution loop for OpenAI, Google, and Anthropic kits.

``MCPAgent`` bridges the gap between an LLM developer kit and a
:class:`~ractogateway.tools.registry.ToolRegistry` (populated from one or
more MCP servers) by running the full agentic loop automatically:

.. code-block:: text

    LLM call
       │
       ├─ finish_reason == "tool_call"  ──►  execute all tool calls
       │                                         │
       │                                         └──► append results as
       │                                              user follow-up
       │
       └─ finish_reason != "tool_call"  ──►  return final LLMResponse

The loop repeats up to *max_turns* times to prevent infinite recursion.

Works with all three provider developer kits — the kit is duck-typed via a
lightweight :class:`_ChatKitProtocol` so no provider package is needed at
import time.

Usage
-----
::

    from ractogateway.openai_developer_kit import OpenAIDeveloperKit
    from ractogateway.mcp import MCPAgent, MCPClientConfig, RactoMCPClient
    from ractogateway._models.chat import ChatConfig

    # Build a ToolRegistry from an MCP server
    config = MCPClientConfig(transport="stdio", command="python",
                             args=["-m", "my_server"])
    registry = RactoMCPClient(config).list_tools_sync()   # one-shot fetch

    # Or build the registry async:
    # async with RactoMCPClient(config) as c:
    #     registry = await c.to_registry()

    kit   = OpenAIDeveloperKit(model="gpt-4o")
    agent = MCPAgent(kit, registry, max_turns=8)

    response = agent.run(
        ChatConfig(user_message="What is the weather in Tokyo and London?")
    )
    print(response.content)

Same code works for Google and Anthropic::

    from ractogateway.google_developer_kit import GoogleDeveloperKit
    kit = GoogleDeveloperKit(model="gemini-2.0-flash")
    agent = MCPAgent(kit, registry)

    from ractogateway.anthropic_developer_kit import AnthropicDeveloperKit
    kit = AnthropicDeveloperKit(model="claude-opus-4-6")
    agent = MCPAgent(kit, registry)
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Protocol, runtime_checkable

from ractogateway._models.chat import ChatConfig, Message, MessageRole
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
from ractogateway.mcp._models import MCPClientConfig
from ractogateway.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Duck-typed kit protocol — no provider SDK imported here
# ---------------------------------------------------------------------------


@runtime_checkable
class _ChatKitProtocol(Protocol):
    """Minimal interface expected of any developer kit."""

    def chat(self, config: ChatConfig) -> LLMResponse:
        """Synchronous chat completion."""
        ...

    async def achat(self, config: ChatConfig) -> LLMResponse:
        """Asynchronous chat completion."""
        ...


# ---------------------------------------------------------------------------
# Tool result formatting
# ---------------------------------------------------------------------------

_TOOL_RESULTS_HEADER = "Tool execution results (use these to answer the user):\n"


def _format_tool_results(
    tool_calls: list[ToolCallResult],
    results: list[str],
) -> str:
    """Format tool call results as a plain-text follow-up user message.

    This provider-agnostic format works with the existing
    :class:`~ractogateway._models.chat.Message` model (``role`` + ``content``)
    across all three developer kits without any schema changes.

    Parameters
    ----------
    tool_calls:
        The tool calls requested by the LLM in the previous turn.
    results:
        String output for each call (parallel ordering with *tool_calls*).

    Returns
    -------
    str
        A human-readable summary that the LLM can consume as a user message.
    """
    lines: list[str] = [_TOOL_RESULTS_HEADER]
    for tc, result in zip(tool_calls, results, strict=True):
        args_repr = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
        lines.append(f"  {tc.name}({args_repr}) → {result}")
    return "\n".join(lines)


def _execute_tool_calls_sync(
    tool_calls: list[ToolCallResult],
    registry: ToolRegistry,
) -> list[str]:
    """Execute tool calls synchronously; return string results."""
    results: list[str] = []
    for tc in tool_calls:
        fn = registry.get_callable(tc.name)
        if fn is None:
            results.append(f"[Error: no callable registered for tool {tc.name!r}]")
            continue
        try:
            raw = fn(**tc.arguments)
            results.append(str(raw))
        except Exception as exc:
            results.append(f"[Error executing {tc.name!r}: {exc}]")
    return results


async def _execute_tool_calls_async(
    tool_calls: list[ToolCallResult],
    registry: ToolRegistry,
) -> list[str]:
    """Execute tool calls (sync or async); return string results."""
    results: list[str] = []
    for tc in tool_calls:
        fn = registry.get_callable(tc.name)
        if fn is None:
            results.append(f"[Error: no callable registered for tool {tc.name!r}]")
            continue
        try:
            if inspect.iscoroutinefunction(fn):
                raw = await fn(**tc.arguments)
            else:
                raw = fn(**tc.arguments)
            results.append(str(raw))
        except Exception as exc:
            results.append(f"[Error executing {tc.name!r}: {exc}]")
    return results


# ---------------------------------------------------------------------------
# MCPAgent
# ---------------------------------------------------------------------------


class MCPAgent:
    """Agentic tool-execution loop compatible with all three developer kits.

    Runs the LLM → tool-call → execute → continue loop automatically,
    returning the final :class:`~ractogateway.adapters.base.LLMResponse` once
    the LLM produces a non-tool response or *max_turns* is reached.

    Parameters
    ----------
    kit:
        Any developer kit with ``chat()`` / ``achat()`` methods:
        :class:`~ractogateway.openai_developer_kit.OpenAIDeveloperKit`,
        :class:`~ractogateway.google_developer_kit.GoogleDeveloperKit`,
        or :class:`~ractogateway.anthropic_developer_kit.AnthropicDeveloperKit`.
    registry:
        Tool registry containing callables for each tool the LLM can call.
        Typically populated via :meth:`RactoMCPClient.to_registry` or
        :meth:`MCPMultiClient.to_registry`.
    max_turns:
        Maximum number of tool-call rounds before the loop stops and
        returns the last response.  Prevents infinite recursion.

    Example
    -------
    ::

        from ractogateway.openai_developer_kit import OpenAIDeveloperKit
        from ractogateway.mcp import MCPAgent, RactoMCPClient, MCPClientConfig
        from ractogateway._models.chat import ChatConfig

        cfg    = MCPClientConfig(transport="stdio", command="python",
                                 args=["-m", "my_server"])
        reg    = RactoMCPClient(cfg).list_tools_sync()
        kit    = OpenAIDeveloperKit(model="gpt-4o")
        agent  = MCPAgent(kit, reg, max_turns=6)
        result = agent.run(ChatConfig(user_message="Search for recent AI papers"))
        print(result.content)
    """

    def __init__(
        self,
        kit: Any,
        registry: ToolRegistry,
        *,
        max_turns: int = 10,
    ) -> None:
        if not isinstance(kit, _ChatKitProtocol):
            raise TypeError(
                f"kit must implement chat() and achat() methods, got {type(kit).__name__!r}.\n"
                "Pass an OpenAIDeveloperKit, GoogleDeveloperKit, or "
                "AnthropicDeveloperKit instance."
            )
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}.")
        self._kit = kit
        self._registry = registry
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_mcp(
        cls,
        kit: Any,
        configs: list[MCPClientConfig],
        *,
        max_turns: int = 10,
    ) -> MCPAgent:
        """Build an agent by fetching tools from one or more MCP servers.

        Opens a **one-shot** connection per config, fetches tools, closes
        the connection, and constructs the agent with the merged registry.

        .. note::

           This is a **sync** classmethod; it uses :func:`asyncio.run` and
           therefore **cannot** be called from within a running event loop.
           In async code, build the registry yourself::

               async with MCPMultiClient(configs) as multi:
                   registry = await multi.to_registry()
               agent = MCPAgent(kit, registry)

        Parameters
        ----------
        kit:
            Any RactoGateway developer kit.
        configs:
            MCP server connection configs.
        max_turns:
            Maximum tool-call rounds.

        Returns
        -------
        MCPAgent
            Ready to call :meth:`run` or :meth:`arun`.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "MCPAgent.from_mcp() cannot be called from a running event loop.\n"
                "Build the registry with 'await MCPMultiClient(configs).to_registry()' "
                "and pass it to MCPAgent() directly."
            )

        from ractogateway.mcp.multi_client import MCPMultiClient
        async def _fetch() -> ToolRegistry:
            async with MCPMultiClient(configs) as multi:
                return await multi.to_registry()

        registry = asyncio.run(_fetch())
        return cls(kit, registry, max_turns=max_turns)

    # ------------------------------------------------------------------
    # Sync agentic loop
    # ------------------------------------------------------------------

    def run(self, config: ChatConfig) -> LLMResponse:
        """Run the agentic loop synchronously.

        Injects the tool registry from this agent into *config* (overriding
        ``config.tools`` if already set).

        Parameters
        ----------
        config:
            Initial chat config.  ``prompt`` must be set here or on the kit.

        Returns
        -------
        LLMResponse
            Final response after tool calls are resolved.
        """
        working_config = config.model_copy(update={"tools": self._registry})
        response = self._kit.chat(working_config)

        for _ in range(self._max_turns):
            if (
                response.finish_reason != FinishReason.TOOL_CALL
                or not response.tool_calls
            ):
                break

            results = _execute_tool_calls_sync(response.tool_calls, self._registry)
            tool_msg = _format_tool_results(response.tool_calls, results)

            new_history = [
                *working_config.history,
                Message(role=MessageRole.ASSISTANT, content=response.content or ""),
                Message(role=MessageRole.USER, content=tool_msg),
            ]
            working_config = working_config.model_copy(
                update={
                    "history": new_history,
                    "user_message": "Continue based on the tool results above.",
                    "tools": self._registry,
                }
            )
            response = self._kit.chat(working_config)

        return response

    # ------------------------------------------------------------------
    # Async agentic loop
    # ------------------------------------------------------------------

    async def arun(self, config: ChatConfig) -> LLMResponse:
        """Run the agentic loop asynchronously.

        Supports async tool callables (``async def``); sync callables are
        called directly.

        Parameters
        ----------
        config:
            Initial chat config.

        Returns
        -------
        LLMResponse
            Final response after tool calls are resolved.
        """
        working_config = config.model_copy(update={"tools": self._registry})
        response = await self._kit.achat(working_config)

        for _ in range(self._max_turns):
            if (
                response.finish_reason != FinishReason.TOOL_CALL
                or not response.tool_calls
            ):
                break

            results = await _execute_tool_calls_async(
                response.tool_calls, self._registry
            )
            tool_msg = _format_tool_results(response.tool_calls, results)

            new_history = [
                *working_config.history,
                Message(role=MessageRole.ASSISTANT, content=response.content or ""),
                Message(role=MessageRole.USER, content=tool_msg),
            ]
            working_config = working_config.model_copy(
                update={
                    "history": new_history,
                    "user_message": "Continue based on the tool results above.",
                    "tools": self._registry,
                }
            )
            response = await self._kit.achat(working_config)

        return response

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    @property
    def registry(self) -> ToolRegistry:
        """The :class:`ToolRegistry` used by this agent."""
        return self._registry

    @property
    def max_turns(self) -> int:
        """Maximum number of tool-call rounds per :meth:`run` call."""
        return self._max_turns

    def __repr__(self) -> str:
        kit_name = type(self._kit).__name__
        return (
            f"MCPAgent(kit={kit_name!r}, "
            f"tools={self._registry.schemas.__len__()}, "
            f"max_turns={self._max_turns})"
        )
