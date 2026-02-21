"""Tests for the ractogateway.mcp package.

All tests are pure-Python: no real MCP server is started.  Provider SDK
packages (openai, anthropic, google-genai) and the ``mcp`` package are NOT
required to run these tests.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from ractogateway.mcp._models import MCPClientConfig, MCPServerConfig, MCPToolResult
from ractogateway.tools.registry import ToolRegistry, ToolSchema

# ---------------------------------------------------------------------------
# ─── Model tests ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    def test_defaults(self) -> None:
        cfg = MCPServerConfig(name="my-server")
        assert cfg.name == "my-server"
        assert cfg.description == ""
        assert cfg.version == "0.1.0"

    def test_frozen(self) -> None:
        cfg = MCPServerConfig(name="srv")
        with pytest.raises(ValidationError):
            cfg.name = "other"  # type: ignore[misc]

    def test_custom_fields(self) -> None:
        cfg = MCPServerConfig(name="x", description="desc", version="2.0.0")
        assert cfg.description == "desc"
        assert cfg.version == "2.0.0"


class TestMCPClientConfig:
    def test_stdio_config(self) -> None:
        cfg = MCPClientConfig(
            transport="stdio",
            command="python",
            args=["-m", "server"],
            env={"KEY": "val"},
        )
        assert cfg.transport == "stdio"
        assert cfg.command == "python"
        assert cfg.args == ["-m", "server"]
        assert cfg.env == {"KEY": "val"}
        assert cfg.url is None

    def test_sse_config(self) -> None:
        cfg = MCPClientConfig(transport="sse", url="http://localhost:8000/sse")
        assert cfg.transport == "sse"
        assert cfg.url == "http://localhost:8000/sse"
        assert cfg.command is None

    def test_streamable_http_config(self) -> None:
        cfg = MCPClientConfig(
            transport="streamable-http", url="http://localhost:9000"
        )
        assert cfg.transport == "streamable-http"

    def test_frozen(self) -> None:
        cfg = MCPClientConfig(transport="stdio", command="python")
        with pytest.raises(ValidationError):
            cfg.command = "node"  # type: ignore[misc]

    def test_invalid_transport(self) -> None:
        with pytest.raises(ValidationError):
            MCPClientConfig(transport="websocket")  # type: ignore[arg-type]


class TestMCPToolResult:
    def test_defaults(self) -> None:
        r = MCPToolResult(content="hello")
        assert r.content == "hello"
        assert r.is_error is False

    def test_error_flag(self) -> None:
        r = MCPToolResult(content="oops", is_error=True)
        assert r.is_error is True


# ---------------------------------------------------------------------------
# ─── RactoMCPServer tests ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestRactoMCPServer:
    def _make_registry(self) -> ToolRegistry:
        registry = ToolRegistry()

        @registry.register
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        @registry.register
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        return registry

    def test_from_registry_populates_schemas(self) -> None:
        from ractogateway.mcp.server import RactoMCPServer

        registry = self._make_registry()
        server = RactoMCPServer.from_registry(registry, name="test-srv")

        assert len(server) == 2
        assert "add" in server
        assert "greet" in server

    def test_tool_names_sorted(self) -> None:
        from ractogateway.mcp.server import RactoMCPServer

        registry = self._make_registry()
        server = RactoMCPServer.from_registry(registry, name="s")
        assert server.tool_names == ["add", "greet"]

    def test_add_tool(self) -> None:
        from ractogateway.mcp.server import RactoMCPServer

        server = RactoMCPServer(name="s")

        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        server.add_tool(multiply)
        assert "multiply" in server

    def test_add_tool_with_name_override(self) -> None:
        from ractogateway.mcp.server import RactoMCPServer

        server = RactoMCPServer(name="s")

        def internal_func(x: int) -> int:
            """Internal."""
            return x

        server.add_tool(internal_func, name="public_func", description="Public API")
        assert "public_func" in server
        assert "internal_func" not in server

    def test_repr(self) -> None:
        from ractogateway.mcp.server import RactoMCPServer

        server = RactoMCPServer(name="demo")
        r = repr(server)
        assert "RactoMCPServer" in r
        assert "demo" in r

    def test_skips_pydantic_model_tools(self) -> None:
        """Tools registered as Pydantic models (no callable) are skipped."""
        from pydantic import BaseModel

        from ractogateway.mcp.server import RactoMCPServer

        class MyModel(BaseModel):
            """A Pydantic model tool."""
            value: int

        registry = ToolRegistry()
        registry.register(MyModel)

        # No callable for Pydantic models — should be silently skipped.
        server = RactoMCPServer.from_registry(registry, name="s")
        assert len(server) == 0

    def test_build_server_uses_our_schema(self) -> None:
        """_build_server must use ToolSchema.to_json_schema(), not re-introspect."""
        from ractogateway.mcp.server import RactoMCPServer

        registry = self._make_registry()
        server = RactoMCPServer.from_registry(registry, name="s")
        # Verify internal schema dict is populated from our ToolSchema
        schema = server._schemas["add"]
        js = schema.to_json_schema()
        assert js["type"] == "object"
        assert "a" in js["properties"]
        assert "b" in js["properties"]


# ---------------------------------------------------------------------------
# ─── _mcp_tool_to_schema tests ──────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestMcpToolToSchema:
    def _make_mock_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> Any:
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.inputSchema = input_schema
        return tool

    def test_basic_conversion(self) -> None:
        from ractogateway.mcp.client import _mcp_tool_to_schema

        mock_tool = self._make_mock_tool(
            name="search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
        schema = _mcp_tool_to_schema(mock_tool)

        assert schema.name == "search"
        assert schema.description == "Search the web"
        assert len(schema.parameters) == 2

        query_param = next(p for p in schema.parameters if p.name == "query")
        assert query_param.type == "string"
        assert query_param.required is True

        limit_param = next(p for p in schema.parameters if p.name == "limit")
        assert limit_param.type == "integer"
        assert limit_param.required is False

    def test_empty_schema(self) -> None:
        from ractogateway.mcp.client import _mcp_tool_to_schema

        mock_tool = self._make_mock_tool("noop", "Does nothing", {})
        schema = _mcp_tool_to_schema(mock_tool)
        assert schema.name == "noop"
        assert schema.parameters == []

    def test_none_description_becomes_empty_string(self) -> None:
        from ractogateway.mcp.client import _mcp_tool_to_schema

        mock_tool = MagicMock()
        mock_tool.name = "tool"
        mock_tool.description = None
        mock_tool.inputSchema = {}
        schema = _mcp_tool_to_schema(mock_tool)
        assert schema.description == ""

    def test_enum_passthrough(self) -> None:
        from ractogateway.mcp.client import _mcp_tool_to_schema

        mock_tool = self._make_mock_tool(
            "sort",
            "Sort items",
            {
                "type": "object",
                "properties": {
                    "order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                    },
                },
                "required": ["order"],
            },
        )
        schema = _mcp_tool_to_schema(mock_tool)
        order_param = schema.parameters[0]
        assert order_param.enum == ["asc", "desc"]


# ---------------------------------------------------------------------------
# ─── _make_remote_callable tests ────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestMakeRemoteCallable:
    def test_raises_in_running_loop(self) -> None:
        from ractogateway.mcp.client import _make_remote_callable

        config = MCPClientConfig(transport="stdio", command="python")
        fn = _make_remote_callable("my_tool", config)

        async def _inner() -> None:
            with pytest.raises(RuntimeError, match="running event loop"):
                fn(x=1)

        asyncio.run(_inner())

    def test_function_name_set(self) -> None:
        from ractogateway.mcp.client import _make_remote_callable

        config = MCPClientConfig(transport="stdio", command="python")
        fn = _make_remote_callable("special_tool", config)
        assert fn.__name__ == "special_tool"


# ---------------------------------------------------------------------------
# ─── RactoMCPClient (mocked) tests ──────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestRactoMCPClientMocked:
    """Tests that mock the mcp package so no real server is needed."""

    def _make_config(self) -> MCPClientConfig:
        return MCPClientConfig(transport="stdio", command="python")

    def test_not_connected_raises(self) -> None:
        from ractogateway.mcp.client import RactoMCPClient

        client = RactoMCPClient(self._make_config())
        with pytest.raises(RuntimeError, match="not connected"):
            client._require_session()
    def test_list_tools_sync_raises_in_loop(self) -> None:
        from ractogateway.mcp.client import RactoMCPClient

        client = RactoMCPClient(self._make_config())

        async def _inner() -> None:
            with pytest.raises(RuntimeError, match="running event loop"):
                client.list_tools_sync()

        asyncio.run(_inner())

    def test_call_tool_sync_raises_in_loop(self) -> None:
        from ractogateway.mcp.client import RactoMCPClient

        client = RactoMCPClient(self._make_config())

        async def _inner() -> None:
            with pytest.raises(RuntimeError, match="running event loop"):
                client.call_tool_sync("tool", {})

        asyncio.run(_inner())

    def test_repr_not_connected(self) -> None:
        from ractogateway.mcp.client import RactoMCPClient

        client = RactoMCPClient(self._make_config())
        assert "connected=False" in repr(client)
        assert "stdio" in repr(client)

    @pytest.mark.asyncio
    async def test_call_tool_parses_result(self) -> None:
        """Verify call_tool correctly extracts text from mcp content blocks."""
        from ractogateway.mcp.client import RactoMCPClient

        config = self._make_config()
        client = RactoMCPClient(config)

        # Manually inject a mock session
        mock_item = MagicMock()
        mock_item.text = "42"
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [mock_item]

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        client._session = mock_session
        result = await client.call_tool("add", {"a": 1, "b": 2})
        assert result.content == "42"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_call_tool_is_error_flag(self) -> None:
        from ractogateway.mcp.client import RactoMCPClient

        config = self._make_config()
        client = RactoMCPClient(config)

        mock_item = MagicMock()
        mock_item.text = "something failed"
        mock_result = MagicMock()
        mock_result.isError = True
        mock_result.content = [mock_item]

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        client._session = mock_session
        result = await client.call_tool("broken_tool", {})
        assert result.is_error is True
        assert result.content == "something failed"

    @pytest.mark.asyncio
    async def test_to_registry_builds_schemas(self) -> None:
        from ractogateway.mcp.client import RactoMCPClient

        config = self._make_config()
        client = RactoMCPClient(config)

        # Build a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }

        mock_list_result = MagicMock()
        mock_list_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_list_result)
        client._session = mock_session
        registry = await client.to_registry()
        assert "search" in registry
        assert registry.get_schema("search") is not None
        assert registry.get_callable("search") is not None


# ---------------------------------------------------------------------------
# ─── MCPMultiClient tests ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestMCPMultiClient:
    def test_requires_at_least_one_config(self) -> None:
        from ractogateway.mcp.multi_client import MCPMultiClient

        with pytest.raises(ValueError, match="at least one"):
            MCPMultiClient([])

    def test_not_connected_raises(self) -> None:
        from ractogateway.mcp.multi_client import MCPMultiClient

        multi = MCPMultiClient(
            [MCPClientConfig(transport="stdio", command="python")]
        )
        with pytest.raises(RuntimeError, match="not connected"):
            multi._require_connected()
    def test_list_tools_sync_raises_in_loop(self) -> None:
        from ractogateway.mcp.multi_client import MCPMultiClient

        multi = MCPMultiClient(
            [MCPClientConfig(transport="stdio", command="python")]
        )

        async def _inner() -> None:
            with pytest.raises(RuntimeError, match="running event loop"):
                multi.list_tools_sync()

        asyncio.run(_inner())

    def test_repr(self) -> None:
        from ractogateway.mcp.multi_client import MCPMultiClient

        multi = MCPMultiClient(
            [
                MCPClientConfig(transport="stdio", command="p"),
                MCPClientConfig(transport="stdio", command="q"),
            ]
        )
        r = repr(multi)
        assert "servers=2" in r

    @pytest.mark.asyncio
    async def test_tool_override_annotates_description(self) -> None:
        """When two servers advertise the same tool, the later one wins and the
        description is annotated with an override note."""
        from ractogateway.mcp.multi_client import MCPMultiClient

        configs = [
            MCPClientConfig(transport="stdio", command="python"),
            MCPClientConfig(transport="stdio", command="python"),
        ]
        multi = MCPMultiClient(configs)

        # Build two mock schemas with the same tool name
        schema_a = ToolSchema(
            name="search",
            description="Server A search",
            parameters=[],
        )
        schema_b = ToolSchema(
            name="search",
            description="Server B search",
            parameters=[],
        )

        # Mock the clients' list_tools
        mock_client_a = AsyncMock()
        mock_client_a.list_tools = AsyncMock(return_value=[schema_a])
        mock_client_a.__aenter__ = AsyncMock(return_value=mock_client_a)
        mock_client_a.__aexit__ = AsyncMock(return_value=None)

        mock_client_b = AsyncMock()
        mock_client_b.list_tools = AsyncMock(return_value=[schema_b])
        mock_client_b.__aenter__ = AsyncMock(return_value=mock_client_b)
        mock_client_b.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "ractogateway.mcp.multi_client.RactoMCPClient",
            side_effect=[mock_client_a, mock_client_b],
        ):
            async with multi:
                tools = await multi.list_tools()

        assert len(tools) == 1
        assert "overrides" in tools[0].description.lower()

    @pytest.mark.asyncio
    async def test_call_tool_routes_to_correct_server(self) -> None:
        """call_tool must forward to the server that owns the tool."""
        from ractogateway.mcp.multi_client import MCPMultiClient

        configs = [
            MCPClientConfig(transport="stdio", command="srv1"),
            MCPClientConfig(transport="stdio", command="srv2"),
        ]
        multi = MCPMultiClient(configs)

        schema_a = ToolSchema(name="math_add", description="Add", parameters=[])
        schema_b = ToolSchema(name="search", description="Search", parameters=[])

        expected_result = MCPToolResult(content="found!")

        mock_client_a = AsyncMock()
        mock_client_a.list_tools = AsyncMock(return_value=[schema_a])
        mock_client_a.__aenter__ = AsyncMock(return_value=mock_client_a)
        mock_client_a.__aexit__ = AsyncMock(return_value=None)

        mock_client_b = AsyncMock()
        mock_client_b.list_tools = AsyncMock(return_value=[schema_b])
        mock_client_b.call_tool = AsyncMock(return_value=expected_result)
        mock_client_b.__aenter__ = AsyncMock(return_value=mock_client_b)
        mock_client_b.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "ractogateway.mcp.multi_client.RactoMCPClient",
            side_effect=[mock_client_a, mock_client_b],
        ):
            async with multi:
                result = await multi.call_tool("search", {"q": "AI"})

        assert result.content == "found!"
        mock_client_b.call_tool.assert_called_once_with("search", {"q": "AI"})

    @pytest.mark.asyncio
    async def test_call_unknown_tool_raises_key_error(self) -> None:
        from ractogateway.mcp.multi_client import MCPMultiClient

        configs = [MCPClientConfig(transport="stdio", command="python")]
        multi = MCPMultiClient(configs)

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("ractogateway.mcp.multi_client.RactoMCPClient", return_value=mock_client):
            async with multi:
                with pytest.raises(KeyError, match="unknown_tool"):
                    await multi.call_tool("unknown_tool", {})


# ---------------------------------------------------------------------------
# ─── MCPAgent tests ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestMCPAgent:
    def _make_registry_with_add(self) -> ToolRegistry:
        registry = ToolRegistry()

        @registry.register
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        return registry

    def _mock_kit(self, responses: list[Any]) -> Any:
        """Create a mock kit that returns a sequence of LLMResponse objects."""
        kit = MagicMock()
        kit.chat = MagicMock(side_effect=responses)
        kit.achat = AsyncMock(side_effect=responses)
        return kit

    def test_rejects_non_kit(self) -> None:
        from ractogateway.mcp.agent import MCPAgent

        with pytest.raises(TypeError, match="chat\\(\\) and achat\\(\\)"):
            MCPAgent(object(), ToolRegistry())

    def test_rejects_zero_max_turns(self) -> None:
        from ractogateway.mcp.agent import MCPAgent

        kit = MagicMock()
        kit.chat = MagicMock()
        kit.achat = AsyncMock()

        with pytest.raises(ValueError, match="max_turns"):
            MCPAgent(kit, ToolRegistry(), max_turns=0)

    def test_repr(self) -> None:
        from ractogateway.mcp.agent import MCPAgent

        kit = MagicMock()
        kit.chat = MagicMock()
        kit.achat = AsyncMock()
        type(kit).__name__ = "OpenAIDeveloperKit"

        agent = MCPAgent(kit, ToolRegistry(), max_turns=5)
        r = repr(agent)
        assert "MCPAgent" in r
        assert "max_turns=5" in r

    def test_run_no_tool_calls(self) -> None:
        """When the LLM returns no tool calls, the loop exits immediately."""
        from ractogateway._models.chat import ChatConfig
        from ractogateway.adapters.base import FinishReason, LLMResponse
        from ractogateway.mcp.agent import MCPAgent

        final = LLMResponse(content="direct answer", finish_reason=FinishReason.STOP)
        kit = self._mock_kit([final])
        registry = ToolRegistry()

        # Need a prompt or default — we skip prompt validation by providing a prompt.
        from ractogateway.prompts.engine import RactoPrompt

        prompt = RactoPrompt(
            role="assistant",
            aim="help",
            constraints=["be accurate"],
            tone="professional",
            output_format="text",
        )
        config = ChatConfig(user_message="hello", prompt=prompt)
        agent = MCPAgent(kit, registry, max_turns=5)
        result = agent.run(config)

        assert result.content == "direct answer"
        kit.chat.assert_called_once()

    def test_run_executes_tool_then_continues(self) -> None:
        """The loop must execute tool calls and feed results back."""
        from ractogateway._models.chat import ChatConfig
        from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
        from ractogateway.mcp.agent import MCPAgent
        from ractogateway.prompts.engine import RactoPrompt

        tool_response = LLMResponse(
            content="",
            finish_reason=FinishReason.TOOL_CALL,
            tool_calls=[ToolCallResult(name="add", arguments={"a": 3, "b": 4})],
        )
        final_response = LLMResponse(
            content="The answer is 7",
            finish_reason=FinishReason.STOP,
        )
        kit = self._mock_kit([tool_response, final_response])
        registry = self._make_registry_with_add()

        prompt = RactoPrompt(
            role="calculator",
            aim="compute",
            constraints=["use tools"],
            tone="neutral",
            output_format="text",
        )
        config = ChatConfig(user_message="What is 3+4?", prompt=prompt)
        agent = MCPAgent(kit, registry, max_turns=5)
        result = agent.run(config)

        assert result.content == "The answer is 7"
        assert kit.chat.call_count == 2

        # The second call must include tool results in the history.
        second_config: ChatConfig = kit.chat.call_args_list[1][0][0]
        assert any(
            "add" in msg.content for msg in second_config.history
        )

    @pytest.mark.asyncio
    async def test_arun_executes_tool_then_continues(self) -> None:
        from ractogateway._models.chat import ChatConfig
        from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
        from ractogateway.mcp.agent import MCPAgent
        from ractogateway.prompts.engine import RactoPrompt

        tool_response = LLMResponse(
            content="",
            finish_reason=FinishReason.TOOL_CALL,
            tool_calls=[ToolCallResult(name="add", arguments={"a": 10, "b": 20})],
        )
        final_response = LLMResponse(
            content="The answer is 30",
            finish_reason=FinishReason.STOP,
        )
        kit = self._mock_kit([tool_response, final_response])
        registry = self._make_registry_with_add()

        prompt = RactoPrompt(
            role="calculator",
            aim="compute",
            constraints=["use tools"],
            tone="neutral",
            output_format="text",
        )
        config = ChatConfig(user_message="What is 10+20?", prompt=prompt)
        agent = MCPAgent(kit, registry, max_turns=5)
        result = await agent.arun(config)

        assert result.content == "The answer is 30"
        assert kit.achat.call_count == 2

    def test_max_turns_stops_loop(self) -> None:
        """After max_turns the loop must return even with ongoing tool calls."""
        from ractogateway._models.chat import ChatConfig
        from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
        from ractogateway.mcp.agent import MCPAgent
        from ractogateway.prompts.engine import RactoPrompt

        always_tool = LLMResponse(
            content="",
            finish_reason=FinishReason.TOOL_CALL,
            tool_calls=[ToolCallResult(name="add", arguments={"a": 1, "b": 1})],
        )
        # Return tool_calls every time — loop must stop at max_turns=2.
        kit = self._mock_kit([always_tool] * 10)
        registry = self._make_registry_with_add()

        prompt = RactoPrompt(
            role="looper",
            aim="loop",
            constraints=["loop forever"],
            tone="neutral",
            output_format="text",
        )
        config = ChatConfig(user_message="loop", prompt=prompt)
        agent = MCPAgent(kit, registry, max_turns=2)
        agent.run(config)

        # Initial call + max_turns follow-up calls = max_turns + 1 total
        assert kit.chat.call_count == 3

    def test_unknown_tool_returns_error_in_result(self) -> None:
        """If the LLM calls a tool that has no callable, the result should
        include an error message rather than raising."""
        from ractogateway.adapters.base import ToolCallResult
        from ractogateway.mcp.agent import _execute_tool_calls_sync

        tool_calls = [ToolCallResult(name="nonexistent", arguments={})]

        results = _execute_tool_calls_sync(tool_calls, ToolRegistry())
        assert len(results) == 1
        assert "no callable" in results[0].lower() or "Error" in results[0]


# ---------------------------------------------------------------------------
# ─── Integration: main package imports ──────────────────────────────────────
# ---------------------------------------------------------------------------


class TestMainPackageExports:
    def test_mcp_symbols_importable_from_root(self) -> None:
        import ractogateway

        assert hasattr(ractogateway, "RactoMCPServer")
        assert hasattr(ractogateway, "RactoMCPClient")
        assert hasattr(ractogateway, "MCPMultiClient")
        assert hasattr(ractogateway, "MCPAgent")
        assert hasattr(ractogateway, "MCPClientConfig")
        assert hasattr(ractogateway, "MCPServerConfig")
        assert hasattr(ractogateway, "MCPToolResult")

    def test_mcp_submodule_importable(self) -> None:
        import ractogateway

        mcp_mod = ractogateway.mcp
        assert hasattr(mcp_mod, "RactoMCPServer")
        assert hasattr(mcp_mod, "RactoMCPClient")
        assert hasattr(mcp_mod, "MCPAgent")

    def test_mcp_in_all(self) -> None:
        import ractogateway

        assert "RactoMCPServer" in ractogateway.__all__
        assert "RactoMCPClient" in ractogateway.__all__
        assert "MCPAgent" in ractogateway.__all__
        assert "mcp" in ractogateway.__all__
