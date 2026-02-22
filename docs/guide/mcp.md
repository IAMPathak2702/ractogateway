# MCP — Model Context Protocol

RactoGateway's `mcp` package lets you expose tools as an MCP server, connect to any MCP server as a client, and run a provider-agnostic agentic loop.

## Expose Tools as an MCP Server

```python
from ractogateway import ToolRegistry
from ractogateway.mcp import RactoMCPServer

registry = ToolRegistry()

@registry.register
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"Sunny in {city}"

server = RactoMCPServer.from_registry(registry, name="weather-tools")
server.run()   # stdio transport — add to claude_desktop_config.json / Cursor
```

## Connect to an MCP Server (Client)

```python
from ractogateway.mcp import RactoMCPClient, MCPClientConfig

cfg = MCPClientConfig(
    transport="stdio",
    command="python",
    args=["-m", "my_server"],
)
registry = RactoMCPClient(cfg).list_tools_sync()
```

SSE and streamable-HTTP transports are also supported:

```python
cfg = MCPClientConfig(transport="sse", url="http://localhost:8001/sse")
```

## Agentic Loop

`MCPAgent` runs the LLM → tool call → execute → continue loop and works identically with all three provider kits:

```python
from ractogateway.mcp import MCPAgent
from ractogateway.openai_developer_kit import OpenAIDeveloperKit
from ractogateway._models.chat import ChatConfig

kit = OpenAIDeveloperKit(model="gpt-4o")
agent = MCPAgent(kit, registry, max_turns=6)
result = agent.run(ChatConfig(user_message="What is the weather in Tokyo?"))
print(result.content)
```

## Aggregate Multiple Servers

```python
from ractogateway.mcp import MCPMultiClient, MCPClientConfig

configs = [
    MCPClientConfig(transport="stdio", command="python", args=["-m", "math_server"]),
    MCPClientConfig(transport="sse",   url="http://localhost:8001/sse"),
]

async with MCPMultiClient(configs) as multi:
    registry = await multi.to_registry()
```

## Installation

```bash
pip install ractogateway[mcp]        # core (stdio + SSE client)
pip install ractogateway[mcp-sse]    # server SSE transport (starlette + uvicorn)
```
