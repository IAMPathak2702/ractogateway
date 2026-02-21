"""RactoMCPServer — expose a ToolRegistry as a Model Context Protocol server.

Uses the low-level ``mcp.server.Server`` API so that our own
``ToolSchema.to_json_schema()`` schemas are forwarded verbatim to every
MCP client — no re-introspection via FastMCP, no drift.

Requires the ``mcp`` package::

    pip install ractogateway[mcp]

SSE transport additionally requires starlette + uvicorn::

    pip install ractogateway[mcp-sse]

Quick start (stdio — for Claude Desktop / subprocess)::

    from ractogateway import ToolRegistry
    from ractogateway.mcp import RactoMCPServer

    registry = ToolRegistry()

    @registry.register
    def search(query: str, limit: int = 5) -> str:
        '''Search the knowledge base.'''
        return f"top {limit} results for {query!r}"

    server = RactoMCPServer.from_registry(registry, name="my-tools")
    server.run()  # blocks; talks MCP via stdin/stdout

Claude Desktop ``claude_desktop_config.json``::

    {
      "mcpServers": {
        "my-tools": {
          "command": "python",
          "args": ["-m", "my_package.server"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, Literal

from ractogateway.tools.registry import ToolRegistry, ToolSchema, _schema_from_function

# ---------------------------------------------------------------------------
# Lazy provider imports
# ---------------------------------------------------------------------------


def _require_mcp_server() -> tuple[Any, Any]:
    """Lazily import ``mcp.server.Server`` and ``mcp.types``."""
    try:
        import mcp.types as mcp_types
        from mcp.server import Server
        return Server, mcp_types
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required for RactoMCPServer.\n"
            "Install it with:  pip install ractogateway[mcp]"
        ) from exc


def _require_mcp_stdio() -> Any:
    """Lazily import the stdio server context manager."""
    try:
        from mcp.server.stdio import stdio_server
        return stdio_server
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required for RactoMCPServer.\n"
            "Install it with:  pip install ractogateway[mcp]"
        ) from exc


def _require_mcp_sse_server() -> tuple[Any, Any, Any, Any, Any]:
    """Lazily import SSE server transport + starlette + uvicorn."""
    try:
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        return SseServerTransport, Starlette, Route, Mount, uvicorn
    except ImportError as exc:
        raise ImportError(
            "SSE transport requires extra packages.\n"
            "Install with:  pip install ractogateway[mcp-sse]"
        ) from exc


# ---------------------------------------------------------------------------
# RactoMCPServer
# ---------------------------------------------------------------------------


class RactoMCPServer:
    """Expose a :class:`~ractogateway.tools.registry.ToolRegistry` (or
    individual functions) as a Model Context Protocol server.

    Supported transports
    --------------------
    * **stdio** *(default)* — standard I/O; ideal for Claude Desktop and
      any subprocess-based MCP client.
    * **sse** — HTTP Server-Sent Events; for remote / browser-based clients.
      Requires ``pip install ractogateway[mcp-sse]``.

    Parameters
    ----------
    name:
        Server name visible to MCP clients.
    description:
        Optional human-readable description.
    version:
        Server version string.

    Example
    -------
    ::

        from ractogateway import ToolRegistry
        from ractogateway.mcp import RactoMCPServer

        registry = ToolRegistry()

        @registry.register
        def add(a: int, b: int) -> int:
            '''Add two integers.'''
            return a + b

        server = RactoMCPServer.from_registry(registry, name="math-tools")
        server.run()  # stdio, blocking
    """

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        version: str = "0.1.0",
    ) -> None:
        self._name = name
        self._description = description
        self._version = version
        # Internal O(1) lookup maps: name → schema / callable
        self._schemas: dict[str, ToolSchema] = {}
        self._callables: dict[str, Callable[..., Any]] = {}

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        registry: ToolRegistry,
        *,
        name: str = "ractogateway-server",
        description: str = "RactoGateway MCP Server",
        version: str = "0.1.0",
    ) -> RactoMCPServer:
        """Build a server from a populated :class:`ToolRegistry`.

        Every registered callable becomes an MCP tool.  Tools registered
        only as Pydantic models (no backing callable) are silently skipped.

        Parameters
        ----------
        registry:
            A ToolRegistry with one or more registered tools.
        name:
            MCP server name shown to clients.
        description:
            Optional server description.
        version:
            Server version string.

        Returns
        -------
        RactoMCPServer
            Ready-to-run server instance.
        """
        server = cls(name=name, description=description, version=version)
        for schema in registry.schemas:
            fn = registry.get_callable(schema.name)
            if fn is None:
                continue
            server._schemas[schema.name] = schema
            server._callables[schema.name] = fn
        return server

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def add_tool(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a single function as an MCP tool.

        The function's type annotations drive the JSON Schema; its docstring
        provides the description (overridable via *description*).

        Parameters
        ----------
        fn:
            The callable to expose.  Both sync and ``async`` functions are
            supported.
        name:
            Override the tool name (defaults to ``fn.__name__``).
        description:
            Override the tool description (defaults to the docstring).
        """
        schema = _schema_from_function(fn)
        if name is not None:
            schema.name = name
        if description is not None:
            schema.description = description
        self._schemas[schema.name] = schema
        self._callables[schema.name] = fn

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(
        self,
        transport: Literal["stdio", "sse"] = "stdio",
        *,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 8000,
    ) -> None:
        """Start the MCP server (blocking).

        Parameters
        ----------
        transport:
            ``"stdio"`` — standard I/O (default; integrates with Claude
            Desktop and subprocess clients).
            ``"sse"`` — HTTP Server-Sent Events (requires
            ``pip install ractogateway[mcp-sse]``).
        host:
            Bind host for SSE transport.
        port:
            Bind port for SSE transport.
        """
        if transport == "stdio":
            asyncio.run(self._run_stdio())
        elif transport == "sse":
            asyncio.run(self._run_sse(host, port))
        else:
            raise ValueError(
                f"Unknown transport {transport!r}. Choose 'stdio' or 'sse'."
            )

    def get_asgi_app(self) -> Any:
        """Return a Starlette ASGI app for SSE transport.

        Use this to mount the MCP server into an **existing** web application
        rather than starting a standalone server with :meth:`run`.

        Requires ``pip install ractogateway[mcp-sse]``.

        Example
        -------
        ::

            import uvicorn
            from ractogateway.mcp import RactoMCPServer

            server = RactoMCPServer.from_registry(registry, name="tools")
            app = server.get_asgi_app()
            uvicorn.run(app, host="0.0.0.0", port=8000)
        """
        sse_transport_cls, starlette_cls, route_cls, mount_cls, _ = (
            _require_mcp_sse_server()
        )
        mcp_server = self._build_server(*_require_mcp_server())
        sse = sse_transport_cls("/messages/")

        async def _handle_sse(request: Any) -> Any:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await mcp_server.run(
                    streams[0],
                    streams[1],
                    mcp_server.create_initialization_options(),
                )

        return starlette_cls(
            routes=[
                route_cls("/sse", endpoint=_handle_sse),
                mount_cls("/messages/", app=sse.handle_post_message),
            ]
        )

    # ------------------------------------------------------------------
    # Internal async runners
    # ------------------------------------------------------------------

    async def _run_stdio(self) -> None:
        """Run with stdio transport (most common)."""
        server_cls, mcp_types = _require_mcp_server()
        stdio_server = _require_mcp_stdio()
        server = self._build_server(server_cls, mcp_types)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    async def _run_sse(self, host: str, port: int) -> None:
        """Run with HTTP SSE transport."""
        sse_transport_cls, starlette_cls, route_cls, mount_cls, uvicorn = (
            _require_mcp_sse_server()
        )
        server_cls, mcp_types = _require_mcp_server()
        mcp_server = self._build_server(server_cls, mcp_types)
        sse = sse_transport_cls("/messages/")

        async def _handle_sse(request: Any) -> Any:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await mcp_server.run(
                    streams[0],
                    streams[1],
                    mcp_server.create_initialization_options(),
                )

        app = starlette_cls(
            routes=[
                route_cls("/sse", endpoint=_handle_sse),
                mount_cls("/messages/", app=sse.handle_post_message),
            ]
        )
        uconfig = uvicorn.Config(app, host=host, port=port, log_level="info")
        userver = uvicorn.Server(uconfig)
        await userver.serve()

    # ------------------------------------------------------------------
    # Core: build the low-level mcp.server.Server instance
    # ------------------------------------------------------------------

    def _build_server(self, server_cls: Any, mcp_types: Any) -> Any:
        """Construct and configure a ``mcp.server.Server`` instance.

        Registers two handlers:

        * ``list_tools`` — returns :class:`~mcp.types.Tool` objects derived
          from our :class:`~ractogateway.tools.registry.ToolSchema` maps.
        * ``call_tool`` — dispatches to the registered callables; supports
          both sync and ``async`` functions transparently.

        Parameters
        ----------
        server_cls:
            The ``mcp.server.Server`` class (passed in from the lazy import).
        mcp_types:
            The ``mcp.types`` module (passed in from the lazy import).

        Returns
        -------
        Any
            Configured ``mcp.server.Server`` instance.
        """
        # Capture snapshots so closures are not affected by later mutations.
        schemas: dict[str, ToolSchema] = dict(self._schemas)
        callables: dict[str, Callable[..., Any]] = dict(self._callables)

        server: Any = server_cls(self._name)

        @server.list_tools()  # type: ignore[untyped-decorator]
        async def _list_tools() -> list[Any]:
            return [
                mcp_types.Tool(
                    name=s.name,
                    description=s.description,
                    inputSchema=s.to_json_schema(),
                )
                for s in schemas.values()
            ]

        @server.call_tool()  # type: ignore[untyped-decorator]
        async def _call_tool(
            name: str,
            arguments: dict[str, Any] | None,
        ) -> list[Any]:
            if name not in callables:
                raise ValueError(
                    f"Unknown tool {name!r}. "
                    f"Available: {sorted(callables)}"
                )
            fn = callables[name]
            args: dict[str, Any] = arguments or {}
            if inspect.iscoroutinefunction(fn):
                result = await fn(**args)
            else:
                result = fn(**args)
            return [mcp_types.TextContent(type="text", text=str(result))]

        return server

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        """Sorted list of registered tool names."""
        return sorted(self._schemas)

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._schemas)

    def __contains__(self, name: str) -> bool:
        """``True`` if *name* is a registered tool."""
        return name in self._schemas

    def __repr__(self) -> str:
        return (
            f"RactoMCPServer(name={self._name!r}, "
            f"tools={self.tool_names})"
        )
