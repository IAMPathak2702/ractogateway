"""Pydantic models for MCP server and client configuration.

These models carry no optional-dependency imports: they are safe to import
even without the ``mcp`` package installed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Configuration for a :class:`RactoMCPServer` instance.

    Parameters
    ----------
    name:
        Server name shown to MCP clients (e.g. Claude Desktop).
    description:
        Optional human-readable description of the server.
    version:
        Server version string (SemVer recommended).
    """

    name: str = Field(..., description="Server name shown to MCP clients.")
    description: str = Field("", description="Optional server description.")
    version: str = Field("0.1.0", description="Server version string (SemVer).")

    model_config = {"frozen": True}


class MCPClientConfig(BaseModel):
    """Configuration for connecting to an MCP server.

    Parameters
    ----------
    transport:
        MCP transport to use.  ``"stdio"`` is standard for subprocess-based
        servers (e.g. Claude Desktop). ``"sse"`` / ``"streamable-http"`` are
        for HTTP-based servers.
    command:
        Executable to launch the server process — **required for stdio**.
    args:
        Command-line arguments passed to *command* (stdio only).
    env:
        Extra environment variables injected into the server process
        (stdio only).  Merged on top of the inherited environment.
    url:
        Server URL — **required for sse / streamable-http**.
        Example: ``"http://localhost:8000/sse"``.
    """

    transport: Literal["stdio", "sse", "streamable-http"] = Field(
        ...,
        description="MCP transport protocol.",
    )
    command: str | None = Field(
        None,
        description="Server process command (stdio transport only).",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Command arguments (stdio transport only).",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables (stdio transport only).",
    )
    url: str | None = Field(
        None,
        description="Server URL (sse / streamable-http transport only).",
    )

    model_config = {"frozen": True}


class MCPToolResult(BaseModel):
    """Result returned from calling a remote MCP tool.

    Parameters
    ----------
    content:
        Text content of the tool response.  Multiple content blocks
        (e.g. from parallel calls) are joined with ``"\\n"``.
    is_error:
        ``True`` when the tool itself reported an error condition.
    """

    content: str = Field(..., description="Text content of the tool response.")
    is_error: bool = Field(False, description="True if the tool reported an error.")
