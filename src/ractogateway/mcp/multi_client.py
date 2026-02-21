"""MCPMultiClient — aggregate tools from multiple MCP servers into one ToolRegistry.

Connects to N servers in parallel, merges their tool schemas, and routes
``call_tool`` requests back to whichever server originally advertised the
tool.  The resulting :class:`~ractogateway.tools.registry.ToolRegistry` is
compatible with **all three** developer kits
(``OpenAIDeveloperKit``, ``GoogleDeveloperKit``, ``AnthropicDeveloperKit``).

Requires the ``mcp`` package::

    pip install ractogateway[mcp]

Example
-------
::

    from ractogateway.mcp import MCPMultiClient, MCPClientConfig

    configs = [
        MCPClientConfig(transport="stdio", command="python",
                        args=["-m", "pkg.math_server"]),
        MCPClientConfig(transport="stdio", command="python",
                        args=["-m", "pkg.search_server"]),
    ]

    async with MCPMultiClient(configs) as multi:
        tools    = await multi.list_tools()
        registry = await multi.to_registry()   # use with any kit
        result   = await multi.call_tool("search", {"query": "AI"})

Sync (one-shot)::

    multi = MCPMultiClient(configs)
    tools = multi.list_tools_sync()
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from ractogateway.mcp._models import MCPClientConfig, MCPToolResult
from ractogateway.mcp.client import (
    RactoMCPClient,
    _make_remote_callable,
)
from ractogateway.tools.registry import ToolRegistry, ToolSchema

# ---------------------------------------------------------------------------
# One-shot async helpers
# ---------------------------------------------------------------------------


async def _one_shot_multi_list(configs: list[MCPClientConfig]) -> list[ToolSchema]:
    """Connect to all servers, merge tools, disconnect."""
    async with MCPMultiClient(configs) as multi:
        return await multi.list_tools()


async def _one_shot_multi_call(
    configs: list[MCPClientConfig],
    name: str,
    arguments: dict[str, Any] | None,
) -> MCPToolResult:
    """Connect to all servers, call tool on the right one, disconnect."""
    async with MCPMultiClient(configs) as multi:
        return await multi.call_tool(name, arguments)


# ---------------------------------------------------------------------------
# MCPMultiClient
# ---------------------------------------------------------------------------


class MCPMultiClient:
    """Connect to **multiple** MCP servers and present them as a single tool surface.

    Tools from all servers are merged into one flat namespace.  If two
    servers advertise the same tool name, the later server's definition
    wins (and a warning is embedded in the tool description noting the
    override).

    Routing is O(1): an internal ``dict[tool_name → server_index]`` maps
    each tool back to its origin server for ``call_tool`` dispatch.

    Parameters
    ----------
    configs:
        One :class:`~ractogateway.mcp._models.MCPClientConfig` per server.
        At least one config is required.
    """

    def __init__(self, configs: list[MCPClientConfig]) -> None:
        if not configs:
            raise ValueError("MCPMultiClient requires at least one MCPClientConfig.")
        self._configs: list[MCPClientConfig] = list(configs)
        # Populated on __aenter__:
        self._clients: list[RactoMCPClient] = []
        self._tool_server_idx: dict[str, int] = {}  # tool_name → client index
        self._schemas: dict[str, ToolSchema] = {}
        self._exit_stack: AsyncExitStack | None = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MCPMultiClient:
        self._exit_stack = AsyncExitStack()
        self._clients = []
        self._tool_server_idx = {}
        self._schemas = {}

        # Open all clients in sequence (each is itself a context manager).
        for idx, config in enumerate(self._configs):
            client = await self._exit_stack.enter_async_context(
                RactoMCPClient(config)
            )
            self._clients.append(client)

            # Fetch and merge tool schemas from this server.
            server_tools = await client.list_tools()
            for raw_schema in server_tools:
                if raw_schema.name in self._tool_server_idx:
                    # Later server overrides; annotate so users are aware.
                    prev_idx = self._tool_server_idx[raw_schema.name]
                    merged_schema: ToolSchema = ToolSchema(
                        name=raw_schema.name,
                        description=(
                            f"{raw_schema.description} "
                            f"[overrides server#{prev_idx}]"
                        ).strip(),
                        parameters=raw_schema.parameters,
                    )
                else:
                    merged_schema = raw_schema
                self._tool_server_idx[merged_schema.name] = idx
                self._schemas[merged_schema.name] = merged_schema

        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._clients = []
            self._tool_server_idx = {}
            self._schemas = {}

    def _require_connected(self) -> None:
        """Raise if the multi-client is not inside an ``async with`` block."""
        if self._exit_stack is None:
            raise RuntimeError(
                "MCPMultiClient is not connected. "
                "Use 'async with MCPMultiClient(configs) as multi:'."
            )

    # ------------------------------------------------------------------
    # Core async API
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[ToolSchema]:
        """Return the merged list of tool schemas from all servers.

        Returns
        -------
        list[ToolSchema]
            Deduplicated (last-server-wins) tool schemas sorted by name.
        """
        self._require_connected()
        return sorted(self._schemas.values(), key=lambda s: s.name)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Call a tool on whichever server originally advertised it.

        Routing is O(1) via the internal ``tool_name → server_index`` map.

        Parameters
        ----------
        name:
            Tool name (must exist in the merged namespace).
        arguments:
            Tool arguments; ``None`` or ``{}`` for parameterless tools.

        Returns
        -------
        MCPToolResult
            Tool output.

        Raises
        ------
        KeyError
            If *name* is not in the merged tool namespace.
        RuntimeError
            If called outside an ``async with`` block.
        """
        self._require_connected()
        if name not in self._tool_server_idx:
            raise KeyError(
                f"Unknown tool {name!r}. "
                f"Available: {sorted(self._tool_server_idx)}"
            )
        idx = self._tool_server_idx[name]
        client = self._clients[idx]
        return await client.call_tool(name, arguments)

    async def to_registry(self) -> ToolRegistry:
        """Return a merged :class:`ToolRegistry` with remote callables.

        Each callable in the registry makes a **fresh** one-shot connection
        to the correct origin server when invoked.  This keeps the registry
        self-contained and usable outside an ``async with`` block.

        Returns
        -------
        ToolRegistry
            Merged registry compatible with all three developer kits.
        """
        self._require_connected()
        registry = ToolRegistry()
        for name, schema in self._schemas.items():
            origin_config = self._configs[self._tool_server_idx[name]]
            fn = _make_remote_callable(name, origin_config)
            registry._tools[name] = schema
            registry._callables[name] = fn
        return registry

    # ------------------------------------------------------------------
    # Sync convenience wrappers
    # ------------------------------------------------------------------

    def list_tools_sync(self) -> list[ToolSchema]:
        """Synchronous wrapper: connect all, list merged tools, disconnect all.

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
                "Use 'async with MCPMultiClient(configs) as multi:' and "
                "'await multi.list_tools()' instead."
            )
        return asyncio.run(_one_shot_multi_list(self._configs))

    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Synchronous wrapper: connect all, call tool, disconnect all.

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
                "Use 'async with MCPMultiClient(configs) as multi:' and "
                "'await multi.call_tool(name, args)' instead."
            )
        return asyncio.run(_one_shot_multi_call(self._configs, name, arguments))

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        """Sorted list of all tool names across all servers."""
        return sorted(self._schemas)

    @property
    def server_count(self) -> int:
        """Number of configured MCP servers."""
        return len(self._configs)

    def __len__(self) -> int:
        """Total number of unique tools across all servers."""
        return len(self._schemas)

    def __contains__(self, name: str) -> bool:
        """``True`` if *name* exists in the merged tool namespace."""
        return name in self._schemas

    def __repr__(self) -> str:
        return (
            f"MCPMultiClient(servers={self.server_count}, "
            f"tools={len(self)})"
        )
