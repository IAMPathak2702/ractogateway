"""RactoGateway MCP — Model Context Protocol server, client, and agentic loop.

This package provides four primary classes:

* :class:`RactoMCPServer` — expose any :class:`~ractogateway.tools.ToolRegistry`
  as an MCP server (stdio or SSE transport).  Works with tools built for any
  of the three provider developer kits.

* :class:`RactoMCPClient` — connect to an MCP server (stdio / SSE /
  streamable-HTTP) and import its tools as a
  :class:`~ractogateway.tools.ToolRegistry` compatible with every kit.

* :class:`MCPMultiClient` — connect to **N** MCP servers simultaneously
  and present their tools as a single merged
  :class:`~ractogateway.tools.ToolRegistry`.

* :class:`MCPAgent` — automatic agentic loop (LLM → tool call → execute →
  continue) that works identically with
  :class:`~ractogateway.openai_developer_kit.OpenAIDeveloperKit`,
  :class:`~ractogateway.google_developer_kit.GoogleDeveloperKit`, and
  :class:`~ractogateway.anthropic_developer_kit.AnthropicDeveloperKit`.

Configuration models:

* :class:`MCPServerConfig` — server name / version metadata.
* :class:`MCPClientConfig` — transport, command / URL, env vars.
* :class:`MCPToolResult` — content + error flag returned by tool calls.

Quick start
-----------

**Server** (expose tools via MCP, works with Claude Desktop / Cursor / …)::

    from ractogateway import ToolRegistry
    from ractogateway.mcp import RactoMCPServer

    registry = ToolRegistry()

    @registry.register
    def get_weather(city: str) -> str:
        '''Return current weather for a city.'''
        return f"Sunny in {city}"

    server = RactoMCPServer.from_registry(registry, name="weather-tools")
    server.run()  # stdio, blocking — add to claude_desktop_config.json

**Client + any developer kit**::

    from ractogateway.mcp import RactoMCPClient, MCPClientConfig, MCPAgent
    from ractogateway.openai_developer_kit import OpenAIDeveloperKit
    from ractogateway._models.chat import ChatConfig

    cfg      = MCPClientConfig(transport="stdio", command="python",
                               args=["-m", "my_server"])
    registry = RactoMCPClient(cfg).list_tools_sync()

    # Works for OpenAI, Google Gemini, and Anthropic Claude:
    kit    = OpenAIDeveloperKit(model="gpt-4o")
    agent  = MCPAgent(kit, registry, max_turns=6)
    result = agent.run(ChatConfig(user_message="What is the weather in Tokyo?"))
    print(result.content)

**Multi-server aggregation**::

    from ractogateway.mcp import MCPMultiClient, MCPClientConfig

    configs = [
        MCPClientConfig(transport="stdio", command="python",
                        args=["-m", "math_server"]),
        MCPClientConfig(transport="sse", url="http://localhost:8001/sse"),
    ]

    async with MCPMultiClient(configs) as multi:
        registry = await multi.to_registry()

Installation
------------
::

    pip install ractogateway[mcp]          # core (stdio + SSE client)
    pip install ractogateway[mcp-sse]      # server SSE transport (+ starlette/uvicorn)
"""

from __future__ import annotations

from ractogateway.mcp._models import MCPClientConfig, MCPServerConfig, MCPToolResult
from ractogateway.mcp.agent import MCPAgent
from ractogateway.mcp.client import RactoMCPClient
from ractogateway.mcp.multi_client import MCPMultiClient
from ractogateway.mcp.server import RactoMCPServer

__all__ = [
    # Config / result models
    "MCPClientConfig",
    "MCPServerConfig",
    "MCPToolResult",
    # Core classes
    "RactoMCPServer",
    "RactoMCPClient",
    "MCPMultiClient",
    "MCPAgent",
]
