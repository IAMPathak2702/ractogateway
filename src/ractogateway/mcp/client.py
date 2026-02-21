"""RactoMCPClient — connect to an MCP server and consume its tools.

Requires the ``mcp`` package::

    pip install ractogateway[mcp]

Usage (async context manager — recommended for long-lived connections)::

    from ractogateway.mcp import RactoMCPClient, MCPClientConfig

    config = MCPClientConfig(
        transport="stdio",
        command="python",
        args=["-m", "my_package.server"],
    )

    async with RactoMCPClient(config) as client:
        tools = await client.list_tools()
        result = await client.call_tool("add", {"a": 1, "b": 2})
        registry = await client.to_registry()   # use with any kit

Usage (sync, one-shot — for scripts / REPLs)::

    client = RactoMCPClient(config)
    tools = client.list_tools_sync()
    result = client.call_tool_sync("add", {"a": 1, "b": 2})

.. note::

   The sync ``*_sync()`` helpers use :func:`asyncio.run` and **cannot** be
   called from within a running event loop (e.g. inside an ``async def`` or
   a Jupyter notebook with ``%autoawait``).  Use the async context manager
   interface in those environments.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

from ractogateway.mcp._models import MCPClientConfig, MCPToolResult
from ractogateway.tools.registry import ParamSchema, ToolRegistry, ToolSchema

# ---------------------------------------------------------------------------
# Lazy provider imports
# ---------------------------------------------------------------------------


def _require_mcp_client() -> tuple[Any, Any, Any]:
    """Lazily import ``ClientSession``, ``StdioServerParameters``, ``stdio_client``."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        return ClientSession, StdioServerParameters, stdio_client
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required for RactoMCPClient.\n"
            "Install it with:  pip install ractogateway[mcp]"
        ) from exc


def _require_mcp_sse_client() -> Any:
    """Lazily import the SSE client context-manager factory."""
    try:
        from mcp.client.sse import sse_client
        return sse_client
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required for SSE client transport.\n"
            "Install it with:  pip install ractogateway[mcp]"
        ) from exc


# ---------------------------------------------------------------------------
# Schema conversion helper
# ---------------------------------------------------------------------------


def _mcp_tool_to_schema(tool: Any) -> ToolSchema:
    """Convert a raw MCP ``Tool`` object to a :class:`ToolSchema`.

    Parses the ``tool.inputSchema`` JSON Schema dict into a list of
    :class:`~ractogateway.tools.registry.ParamSchema` objects, preserving
    types, descriptions, required flags, enums, and defaults.

    Parameters
    ----------
    tool:
        An ``mcp.types.Tool`` instance returned by ``session.list_tools()``.

    Returns
    -------
    ToolSchema
        Provider-agnostic canonical tool representation.
    """
    params: list[ParamSchema] = []
    input_schema: dict[str, Any] = getattr(tool, "inputSchema", {}) or {}
    properties: dict[str, Any] = input_schema.get("properties", {})
    required_set: set[str] = set(input_schema.get("required", []))

    for param_name, param_info in properties.items():
        params.append(
            ParamSchema(
                name=param_name,
                type=param_info.get("type", "string"),
                description=param_info.get("description", ""),
                required=param_name in required_set,
                enum=param_info.get("enum"),
                default=param_info.get("default"),
            )
        )

    return ToolSchema(
        name=tool.name,
        description=getattr(tool, "description", "") or "",
        parameters=params,
    )


# ---------------------------------------------------------------------------
# Remote callable factory
# ---------------------------------------------------------------------------


def _make_remote_callable(
    tool_name: str,
    config: MCPClientConfig,
) -> Callable[..., str]:
    """Return a sync callable that makes a **one-shot** MCP tool call.

    Each invocation opens a fresh connection, calls the tool, and closes
    the connection.  For high-throughput use cases, hold a
    :class:`RactoMCPClient` context manager open and call
    :meth:`~RactoMCPClient.call_tool` directly.

    Parameters
    ----------
    tool_name:
        Name of the remote tool.
    config:
        Connection configuration for the MCP server hosting the tool.

    Returns
    -------
    Callable[..., str]
        A sync callable with ``**kwargs`` signature that returns the tool's
        text output.
    """

    async def _call_async(**kwargs: Any) -> str:
        async with RactoMCPClient(config) as client:
            result = await client.call_tool(tool_name, kwargs)
            return result.content

    def _call_sync(**kwargs: Any) -> str:
        # Guard: asyncio.run() fails when a loop is already running.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # No running loop — safe to call asyncio.run().
        else:
            raise RuntimeError(
                f"Cannot call remote MCP tool {tool_name!r} synchronously "
                "from within a running event loop.\n"
                "Use 'async with RactoMCPClient(config) as client:' and "
                "'await client.call_tool(name, args)' instead."
            )
        return asyncio.run(_call_async(**kwargs))

    _call_sync.__name__ = tool_name
    _call_sync.__doc__ = f"Remote MCP tool: {tool_name!r}"
    return _call_sync


# ---------------------------------------------------------------------------
# One-shot async helpers (used by sync wrappers)
# ---------------------------------------------------------------------------


async def _one_shot_list_tools(config: MCPClientConfig) -> list[ToolSchema]:
    """Open connection, list tools, close connection."""
    async with RactoMCPClient(config) as client:
        return await client.list_tools()


async def _one_shot_call_tool(
    config: MCPClientConfig,
    name: str,
    arguments: dict[str, Any] | None,
) -> MCPToolResult:
    """Open connection, call tool, close connection."""
    async with RactoMCPClient(config) as client:
        return await client.call_tool(name, arguments)


# ---------------------------------------------------------------------------
# RactoMCPClient
# ---------------------------------------------------------------------------


class RactoMCPClient:
    """Connect to an MCP server and consume its tools as
    :class:`~ractogateway.tools.registry.ToolSchema` objects.

    This is an **async context manager**.  Keep it alive to reuse the
    underlying connection for multiple tool calls (O(1) per call after
    connection setup).  For single calls from synchronous code, use the
    ``*_sync()`` convenience methods.

    Parameters
    ----------
    config:
        Connection configuration (transport, command / URL, env, …).

    Example — async (recommended)
    ------------------------------
    ::

        config = MCPClientConfig(transport="stdio", command="python",
                                 args=["-m", "my_server"])

        async with RactoMCPClient(config) as client:
            # Reuse this connection for all calls.
            tools   = await client.list_tools()
            result  = await client.call_tool("search", {"query": "AI"})
            registry = await client.to_registry()

    Example — sync one-shot
    -----------------------
    ::

        client = RactoMCPClient(config)
        tools = client.list_tools_sync()
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self._config = config
        self._session: Any | None = None
        self._exit_stack: AsyncExitStack | None = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> RactoMCPClient:
        await self._connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._disconnect()

    async def _connect(self) -> None:
        """Open transport and initialise the MCP ``ClientSession``."""
        client_session_cls, stdio_params_cls, stdio_client = _require_mcp_client()
        self._exit_stack = AsyncExitStack()

        if self._config.transport == "stdio":
            if self._config.command is None:
                raise ValueError(
                    "MCPClientConfig.command is required for stdio transport."
                )
            params = stdio_params_cls(
                command=self._config.command,
                args=list(self._config.args),
                # Pass None instead of an empty dict — some mcp versions
                # only accept None or a non-empty mapping.
                env=dict(self._config.env) if self._config.env else None,
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )

        elif self._config.transport in ("sse", "streamable-http"):
            if self._config.url is None:
                raise ValueError(
                    f"MCPClientConfig.url is required for "
                    f"{self._config.transport!r} transport."
                )
            sse_client = _require_mcp_sse_client()
            read, write = await self._exit_stack.enter_async_context(
                sse_client(self._config.url)
            )

        else:
            raise ValueError(
                f"Unknown transport {self._config.transport!r}. "
                "Choose 'stdio', 'sse', or 'streamable-http'."
            )

        session: Any = await self._exit_stack.enter_async_context(
            client_session_cls(read, write)
        )
        await session.initialize()
        self._session = session

    async def _disconnect(self) -> None:
        """Close the MCP ``ClientSession`` and transport."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

    def _require_session(self) -> Any:
        """Return the active session or raise a helpful ``RuntimeError``."""
        if self._session is None:
            raise RuntimeError(
                "RactoMCPClient is not connected. "
                "Use 'async with RactoMCPClient(config) as client:' "
                "or call one of the *_sync() convenience methods."
            )
        return self._session

    # ------------------------------------------------------------------
    # Core async API
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[ToolSchema]:
        """List all tools exposed by the MCP server.

        Returns
        -------
        list[ToolSchema]
            Provider-agnostic tool schemas — ready to be registered in any
            :class:`~ractogateway.tools.registry.ToolRegistry` or passed
            directly to a developer kit via ``ChatConfig(tools=…)``.

        Raises
        ------
        RuntimeError
            If called outside an ``async with`` block.
        """
        session = self._require_session()
        result = await session.list_tools()
        return [_mcp_tool_to_schema(t) for t in result.tools]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Call a remote MCP tool.

        Parameters
        ----------
        name:
            Tool name (must exist on the server).
        arguments:
            Keyword arguments to pass to the tool.  Pass ``None`` or ``{}``
            for tools with no parameters.

        Returns
        -------
        MCPToolResult
            ``content`` contains all text blocks joined by ``"\\n"``.
            ``is_error`` is ``True`` when the server signals a tool error.

        Raises
        ------
        RuntimeError
            If called outside an ``async with`` block.
        """
        session = self._require_session()
        raw = await session.call_tool(name, arguments or {})

        is_error: bool = bool(getattr(raw, "isError", False))
        texts: list[str] = []
        for item in getattr(raw, "content", []):
            text: str | None = getattr(item, "text", None)
            if text is not None:
                texts.append(text)

        return MCPToolResult(content="\n".join(texts), is_error=is_error)

    async def to_registry(self) -> ToolRegistry:
        """Return a :class:`ToolRegistry` populated with all server tools.

        Each callable in the registry makes a **fresh** one-shot MCP
        connection when invoked.  This keeps the returned registry
        self-contained and usable outside an ``async with`` block.

        For high-throughput usage, hold the :class:`RactoMCPClient` context
        manager alive and call :meth:`call_tool` directly.

        Returns
        -------
        ToolRegistry
            Registry compatible with all three developer kits via
            ``ChatConfig(tools=registry)``.
        """
        schemas = await self.list_tools()
        registry = ToolRegistry()
        for schema in schemas:
            fn = _make_remote_callable(schema.name, self._config)
            # Access private dicts directly — same package, intentional.
            registry._tools[schema.name] = schema
            registry._callables[schema.name] = fn
        return registry

    # ------------------------------------------------------------------
    # Sync convenience wrappers (one-shot: connect → call → disconnect)
    # ------------------------------------------------------------------

    def list_tools_sync(self) -> list[ToolSchema]:
        """Synchronous wrapper: connect, list tools, disconnect.

        Returns
        -------
        list[ToolSchema]
            All tool schemas exposed by the server.

        Raises
        ------
        RuntimeError
            If called from within a running event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "list_tools_sync() cannot be called from a running event loop.\n"
                "Use 'async with RactoMCPClient(config) as client:' and "
                "'await client.list_tools()' instead."
            )
        return asyncio.run(_one_shot_list_tools(self._config))

    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Synchronous wrapper: connect, call tool, disconnect.

        Parameters
        ----------
        name:
            Tool name.
        arguments:
            Tool arguments.

        Returns
        -------
        MCPToolResult
            Tool output.

        Raises
        ------
        RuntimeError
            If called from within a running event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "call_tool_sync() cannot be called from a running event loop.\n"
                "Use 'async with RactoMCPClient(config) as client:' and "
                "'await client.call_tool(name, args)' instead."
            )
        return asyncio.run(_one_shot_call_tool(self._config, name, arguments))

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        connected = self._session is not None
        return (
            f"RactoMCPClient(transport={self._config.transport!r}, "
            f"connected={connected})"
        )
