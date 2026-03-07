"""Models for AgentPipeline."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StopReason(str, Enum):
    """Why the agent loop terminated."""

    FINISHED = "finished"          # finish() tool was called - task complete
    MAX_STEPS = "max_steps"        # hit the max_steps hard cap
    ERROR = "error"                # unrecoverable exception (safe_mode=True only)
    CIRCUIT_BREAK = "circuit_break"  # too many consecutive tool errors → forced finish


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


class AgentUsage(BaseModel):
    """Token and step accounting across the full agent run."""

    total_input_tokens: int = 0
    """Cumulative input tokens across all LLM calls."""

    total_output_tokens: int = 0
    """Cumulative output tokens across all LLM calls."""

    steps_taken: int = 0
    """Number of reasoning+action steps executed."""

    tools_called: int = 0
    """Number of non-finish tool invocations."""

    @property
    def total_tokens(self) -> int:
        """Sum of all input and output tokens."""
        return self.total_input_tokens + self.total_output_tokens


# ---------------------------------------------------------------------------
# Step and result models
# ---------------------------------------------------------------------------


class AgentStep(BaseModel):
    """One reasoning + action step in the ReAct loop."""

    step_num: int
    """1-based position in the execution sequence."""

    thought: str | None = None
    """The LLM's reasoning text before selecting a tool."""

    tool_name: str
    """Name of the tool that was invoked."""

    tool_input: dict[str, Any] = Field(default_factory=dict)
    """Arguments passed to the tool as a flat dict."""

    observation: str = ""
    """The tool's return value (truncated to 4000 chars when long)."""

    duration_ms: float = 0.0
    """Wall-clock time for tool execution in milliseconds."""

    is_finish: bool = False
    """True when this step invoked the finish() termination tool."""


class AgentResult(BaseModel):
    """Full output of an :class:`AgentPipeline` run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    goal: str
    """The original user goal / task string."""

    final_answer: str | None = None
    """The agent's answer when finish() was called (None on max_steps/error)."""

    steps: list[AgentStep] = Field(default_factory=list)
    """All reasoning+action steps in execution order."""

    stop_reason: StopReason = StopReason.FINISHED
    """Why the loop terminated."""

    usage: AgentUsage = Field(default_factory=AgentUsage)
    """Token and step counts."""

    error: str | None = None
    """Exception message when stop_reason is ERROR."""

    parsed_output: Any = Field(default=None, exclude=True)
    """Validated Pydantic model instance when ``response_format`` was passed to ``run()``.
    Not included in JSON serialisation; call ``result.parsed_output.model_dump()`` manually."""

    # ── Convenience accessors ────────────────────────────────────────────────

    def get_tool_calls(self) -> list[tuple[str, dict[str, Any]]]:
        """Return (tool_name, tool_input) for every non-finish step."""
        return [(s.tool_name, s.tool_input) for s in self.steps if not s.is_finish]

    def get_observations(self) -> list[str]:
        """Return all tool observations in execution order."""
        return [s.observation for s in self.steps if not s.is_finish]

    def succeeded(self) -> bool:
        """True when the agent reached a final answer."""
        return self.stop_reason == StopReason.FINISHED and self.final_answer is not None

    # ── Export helpers ───────────────────────────────────────────────────────

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str | None:
        """Serialise to JSON. Returns string when *path* is ``None``."""
        data = self.model_dump()
        if self.parsed_output is not None:
            data["parsed_output"] = (
                self.parsed_output.model_dump()
                if hasattr(self.parsed_output, "model_dump")
                else self.parsed_output
            )
        text = json.dumps(data, indent=indent, default=str)
        if path:
            Path(path).write_text(text, encoding="utf-8")
            return None
        return text

    def to_markdown(self, path: str | None = None) -> str | None:
        """Build a step-by-step Markdown trace. Returns string when *path* is ``None``."""
        lines: list[str] = [
            "# Agent Run\n",
            f"**Goal:** {self.goal}\n",
            f"**Status:** `{self.stop_reason.value}` | "
            f"**Steps:** {len(self.steps)} | "
            f"**Tokens:** {self.usage.total_tokens}\n",
        ]
        for step in self.steps:
            tag = "FINISH" if step.is_finish else f"Step {step.step_num}"
            lines.append(f"---\n\n## {tag}: `{step.tool_name}`")
            if step.thought:
                lines.append(f"\n**Thought:** {step.thought}\n")
            if step.tool_input:
                lines.append(
                    f"\n**Input:**\n```json\n{json.dumps(step.tool_input, indent=2)}\n```"
                )
            if step.observation and not step.is_finish:
                lines.append(
                    f"\n**Observation:** ({step.duration_ms:.0f} ms)\n"
                    f"```\n{step.observation}\n```\n"
                )
        if self.final_answer:
            lines.append(f"---\n\n## Final Answer\n\n{self.final_answer}\n")
        if self.parsed_output is not None:
            dumped = (
                self.parsed_output.model_dump()
                if hasattr(self.parsed_output, "model_dump")
                else self.parsed_output
            )
            lines.append(
                f"---\n\n## Structured Output\n\n"
                f"```json\n{json.dumps(dumped, indent=2, default=str)}\n```\n"
            )
        if self.error:
            lines.append(f"---\n\n## Error\n\n```\n{self.error}\n```\n")

        text = "\n".join(lines)
        if path:
            Path(path).write_text(text, encoding="utf-8")
            return None
        return text


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentRateLimitExceededError(RuntimeError):
    """Raised when the rate limiter blocks an agent run."""
