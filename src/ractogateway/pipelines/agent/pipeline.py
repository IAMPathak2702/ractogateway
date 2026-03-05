"""AgentPipeline - autonomous ReAct agent with pluggable tools.

Implements the Reason+Act (ReAct) loop:

  1. The LLM receives a system prompt listing all registered tools and a
     growing conversation transcript (goal + previous steps).
  2. The LLM responds with a single JSON object:
       {"thought": "...", "tool_name": "...", "tool_input": {...}}
  3. The tool is executed; its result (the *observation*) is appended to
     the transcript.
  4. The loop repeats until the LLM calls ``finish(answer=...)`` or
     ``max_steps`` is reached.

Usage::

    from ractogateway.openai_developer_kit import Chat
    from ractogateway.pipelines.agent import AgentPipeline

    def get_weather(city: str) -> str:
        \"\"\"Return the current weather for a city.\"\"\"
        return f"Sunny, 22 C in {city}"

    agent = AgentPipeline(
        kit=Chat(model="gpt-4o"),
        tools=[get_weather],
        max_steps=6,
        safe_mode=True,
    )

    result = agent.run("What is the weather in Paris and London?")
    print(result.final_answer)
    print(result.to_markdown())

    # Async variant (FastAPI / async servers):
    result = await agent.arun("What is the weather in Tokyo?")
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from ._executor import (
    FINISH_TOOL,
    ToolExecutor,
    make_finish_tool,
    make_http_tool,
    make_memory_tools,
    make_rag_tool,
    make_sql_tool,
)
from ._models import (
    AgentRateLimitExceededError,
    AgentResult,
    AgentStep,
    AgentUsage,
    StopReason,
)

# Sentinel for "not provided by caller"
_UNSET = object()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an autonomous AI agent. You solve tasks step-by-step by calling the \
tools listed below. Think carefully before each action.

AVAILABLE TOOLS:
{tool_descriptions}

STRICT RESPONSE FORMAT - output ONLY a single valid JSON object, nothing else:
{{"thought": "<your reasoning>", "tool_name": "<tool_name>", "tool_input": {{<key-value args>}}}}

RULES:
1. Respond with exactly ONE JSON object per turn - no extra text, no markdown.
2. Always include "thought", "tool_name", and "tool_input" keys.
3. To finish, call the finish tool:
   {{"thought": "I have the answer", "tool_name": "finish", "tool_input": {{"answer": "<full answer>"}}}}
4. If a tool returns an ERROR, adjust your approach and try again.
5. Never invent data - only use facts from tool observations.
6. The "thought" field is your private reasoning - be concise.
{extra_rules}"""


def _build_system_prompt(executor: ToolExecutor, extra_rules: str = "") -> str:
    extra = f"7. {extra_rules}" if extra_rules else ""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=executor.describe_all(),
        extra_rules=extra,
    )


# ---------------------------------------------------------------------------
# Conversation transcript builder
# ---------------------------------------------------------------------------


def _build_transcript(goal: str, steps: list[AgentStep]) -> str:
    """Build the rolling conversation transcript passed to the LLM each step."""
    lines: list[str] = [f"TASK: {goal}\n"]
    for step in steps:
        action = json.dumps(
            {
                "thought": step.thought,
                "tool_name": step.tool_name,
                "tool_input": step.tool_input,
            }
        )
        lines.append(f"Assistant: {action}")
        lines.append(f"Observation: {step.observation}\n")
    lines.append("Your next action (JSON only):")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM response parser
# ---------------------------------------------------------------------------


def _parse_response(
    text: str,
) -> tuple[str | None, str, dict[str, Any]]:
    """Parse the LLM text into ``(thought, tool_name, tool_input)``.

    Handles pure JSON, JSON inside markdown code fences, and malformed
    responses (falls back to ``finish`` with the raw text as the answer).
    """
    raw = text.strip()

    # Strip markdown code fences  ``` or ```json
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Extract the first {...} block (handles trailing text / extra whitespace)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        data = json.loads(raw)
        thought: str | None = data.get("thought") or None
        tool_name = str(data.get("tool_name", FINISH_TOOL)).strip()
        tool_input = data.get("tool_input", {})
        if not isinstance(tool_input, dict):
            tool_input = {"value": str(tool_input)}
        return thought, tool_name, tool_input
    except (json.JSONDecodeError, ValueError):
        # Graceful fallback: treat the whole response as the final answer
        return None, FINISH_TOOL, {"answer": text}


# ---------------------------------------------------------------------------
# AgentPipeline
# ---------------------------------------------------------------------------


class AgentPipeline:
    """Autonomous ReAct agent with sync ``run()`` and async ``arun()``.

    Parameters
    ----------
    kit:
        Any RactoGateway developer kit (must support ``chat()`` and
        ``achat()`` methods).
    tools:
        List of plain callables or ``@tool``-decorated functions to register.
        Each is registered by its ``__name__`` (or ``__tool_name__`` if set
        by the ``@tool`` decorator).
    rag_pipeline:
        Optional ``RactoRAG`` instance - auto-registers a ``rag_search`` tool.
    sql_pipeline:
        Optional ``SQLAnalystPipeline`` - auto-registers a ``sql_query`` tool.
    enable_http:
        When ``True``, registers an ``http_get`` tool that fetches URLs via
        ``httpx`` (requires ``pip install ractogateway[pipelines-agent-http]``).
    agent_memory:
        Any dict-like or object with ``get``/``set`` methods.  Auto-registers
        ``memory_read`` and ``memory_write`` tools.
    max_steps:
        Hard cap on tool calls before the loop stops with
        :attr:`StopReason.MAX_STEPS`.
    system_prompt:
        Fully override the auto-generated system prompt (advanced).
    extra_rules:
        Append an extra rule line to the default system prompt.
    safe_mode:
        Catch all exceptions and surface them in ``result.error`` instead of
        re-raising.
    tracer:
        Optional :class:`ractogateway.telemetry.RactoTracer`.
    metrics:
        Optional :class:`ractogateway.telemetry.GatewayMetricsMiddleware`.
    rate_limiter:
        Duck-typed rate limiter with ``check_and_consume(user_id, tokens)``
        and ``get_remaining(user_id)`` methods.
    user_id:
        Default user identifier passed to the rate limiter.
    """

    def __init__(
        self,
        kit: Any,
        *,
        tools: list[Callable[..., Any]] | None = None,
        rag_pipeline: Any = None,
        sql_pipeline: Any = None,
        enable_http: bool = False,
        agent_memory: Any = None,
        max_steps: int = 10,
        system_prompt: str | None = None,
        extra_rules: str = "",
        safe_mode: bool = False,
        tracer: Any = None,
        metrics: Any = None,
        rate_limiter: Any = None,
        user_id: str = "default",
    ) -> None:
        self._kit = kit
        self._max_steps = max_steps
        self._safe_mode = safe_mode
        self._tracer = tracer
        self._metrics = metrics
        self._rate_limiter = rate_limiter
        self._user_id = user_id

        # ── Build tool registry ───────────────────────────────────────────
        tool_map: dict[str, Callable[..., Any]] = {}

        # 1. Finish — always present
        fname, ffn = make_finish_tool()
        tool_map[fname] = ffn

        # 2. RAG search
        if rag_pipeline is not None:
            tname, tfn = make_rag_tool(rag_pipeline)
            tool_map[tname] = tfn

        # 3. SQL query
        if sql_pipeline is not None:
            tname, tfn = make_sql_tool(sql_pipeline)
            tool_map[tname] = tfn

        # 4. HTTP fetch (opt-in)
        if enable_http:
            tname, tfn = make_http_tool()
            tool_map[tname] = tfn

        # 5. Memory read/write
        if agent_memory is not None:
            for tname, tfn in make_memory_tools(agent_memory):
                tool_map[tname] = tfn

        # 6. User-supplied tools (last, so they can shadow built-ins)
        for fn in tools or []:
            name: str = getattr(fn, "__tool_name__", None) or fn.__name__
            tool_map[name] = fn

        self._executor = ToolExecutor(tool_map)

        # ── System prompt ─────────────────────────────────────────────────
        self._system_prompt = (
            system_prompt
            if system_prompt is not None
            else _build_system_prompt(self._executor, extra_rules)
        )

        # ── RactoPrompt for ChatConfig ─────────────────────────────────────
        # Encode the full system prompt into the aim field so the kit's
        # compiled system message carries all tool/rule content.
        from ractogateway.prompts.engine import RactoPrompt

        self._prompt = RactoPrompt(
            role="You are an autonomous AI agent.",
            aim=self._system_prompt,
            constraints=["Follow all instructions in the aim section exactly."],
            tone="Systematic, precise, and tool-focused.",
            output_format=(
                'Single valid JSON: {"thought": "...", "tool_name": "...", "tool_input": {...}}'
            ),
            anti_hallucination=False,
        )

    # ── Rate limiter ─────────────────────────────────────────────────────────

    def _check_rate_limit(self, user_id: str) -> None:
        if self._rate_limiter is None:
            return
        allowed = self._rate_limiter.check_and_consume(user_id, 1)
        if not allowed:
            raise AgentRateLimitExceededError(
                f"Rate limit exceeded for user '{user_id}'. "
                f"Remaining: {self._rate_limiter.get_remaining(user_id)}"
            )

    # ── Public sync API ──────────────────────────────────────────────────────

    def run(
        self,
        goal: str,
        *,
        max_steps: Any = _UNSET,
        user_id: Any = _UNSET,
        session_id: str | None = None,
    ) -> AgentResult:
        """Run the agent synchronously until it finishes or hits ``max_steps``.

        Parameters
        ----------
        goal:
            The task or question the agent should solve.
        max_steps:
            Per-call override for the constructor ``max_steps``.
        user_id:
            Per-call override for the rate-limiter user identifier.
        session_id:
            Optional session tag passed to tracer spans (informational).

        Returns
        -------
        AgentResult
            Contains the final answer, all steps, stop reason, and token usage.
        """
        uid = self._user_id if user_id is _UNSET else user_id
        steps_cap = self._max_steps if max_steps is _UNSET else int(max_steps)

        if self._safe_mode:
            try:
                return self._run_loop(goal, steps_cap, uid, session_id)
            except Exception as exc:
                return AgentResult(
                    goal=goal,
                    stop_reason=StopReason.ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                )
        return self._run_loop(goal, steps_cap, uid, session_id)

    # ── Public async API ─────────────────────────────────────────────────────

    async def arun(
        self,
        goal: str,
        *,
        max_steps: Any = _UNSET,
        user_id: Any = _UNSET,
        session_id: str | None = None,
    ) -> AgentResult:
        """Async variant of :meth:`run`.

        Uses ``kit.achat()`` for LLM calls and runs sync tools in a thread
        pool so the event loop is never blocked.

        Parameters
        ----------
        goal:
            The task or question the agent should solve.
        max_steps:
            Per-call override for the constructor ``max_steps``.
        user_id:
            Per-call override for the rate-limiter user identifier.
        session_id:
            Optional session tag for tracing.
        """
        uid = self._user_id if user_id is _UNSET else user_id
        steps_cap = self._max_steps if max_steps is _UNSET else int(max_steps)

        if self._safe_mode:
            try:
                return await self._arun_loop(goal, steps_cap, uid, session_id)
            except Exception as exc:
                return AgentResult(
                    goal=goal,
                    stop_reason=StopReason.ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                )
        return await self._arun_loop(goal, steps_cap, uid, session_id)

    # ── Sync execution loop ───────────────────────────────────────────────────

    def _run_loop(
        self,
        goal: str,
        max_steps: int,
        user_id: str,
        session_id: str | None,
    ) -> AgentResult:
        from ractogateway._models.chat import ChatConfig

        self._check_rate_limit(user_id)

        usage = AgentUsage()
        steps: list[AgentStep] = []

        for step_num in range(1, max_steps + 1):
            transcript = _build_transcript(goal, steps)
            chat_config = ChatConfig(user_message=transcript, prompt=self._prompt)
            response = self._kit.chat(chat_config)

            # Accumulate token usage
            if hasattr(response, "usage") and isinstance(response.usage, dict):
                raw: dict[str, Any] = response.usage
                in_tok: int = int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0)
                out_tok: int = int(raw.get("output_tokens") or raw.get("completion_tokens") or 0)
                usage.total_input_tokens += in_tok
                usage.total_output_tokens += out_tok
            usage.steps_taken = step_num

            thought, tool_name, tool_input = _parse_response(response.text)

            # ── Finish ──────────────────────────────────────────────────────
            if tool_name == FINISH_TOOL:
                answer = tool_input.get("answer", response.text)
                steps.append(
                    AgentStep(
                        step_num=step_num,
                        thought=thought,
                        tool_name=FINISH_TOOL,
                        tool_input=tool_input,
                        observation="",
                        is_finish=True,
                    )
                )
                return AgentResult(
                    goal=goal,
                    final_answer=str(answer),
                    steps=steps,
                    stop_reason=StopReason.FINISHED,
                    usage=usage,
                )

            # ── Execute tool ────────────────────────────────────────────────
            observation, duration_ms = self._executor.execute(tool_name, tool_input)
            usage.tools_called += 1

            steps.append(
                AgentStep(
                    step_num=step_num,
                    thought=thought,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    observation=observation,
                    duration_ms=duration_ms,
                )
            )

        # ── Max steps reached ───────────────────────────────────────────────
        return AgentResult(
            goal=goal,
            final_answer=None,
            steps=steps,
            stop_reason=StopReason.MAX_STEPS,
            usage=usage,
        )

    # ── Async execution loop ──────────────────────────────────────────────────

    async def _arun_loop(
        self,
        goal: str,
        max_steps: int,
        user_id: str,
        session_id: str | None,
    ) -> AgentResult:
        from ractogateway._models.chat import ChatConfig

        self._check_rate_limit(user_id)

        usage = AgentUsage()
        steps: list[AgentStep] = []

        for step_num in range(1, max_steps + 1):
            transcript = _build_transcript(goal, steps)
            chat_config = ChatConfig(user_message=transcript, prompt=self._prompt)
            # Async LLM call
            response = await self._kit.achat(chat_config)

            if hasattr(response, "usage") and isinstance(response.usage, dict):
                raw: dict[str, Any] = response.usage
                in_tok: int = int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0)
                out_tok: int = int(raw.get("output_tokens") or raw.get("completion_tokens") or 0)
                usage.total_input_tokens += in_tok
                usage.total_output_tokens += out_tok
            usage.steps_taken = step_num

            thought, tool_name, tool_input = _parse_response(response.text)

            # ── Finish ──────────────────────────────────────────────────────
            if tool_name == FINISH_TOOL:
                answer = tool_input.get("answer", response.text)
                steps.append(
                    AgentStep(
                        step_num=step_num,
                        thought=thought,
                        tool_name=FINISH_TOOL,
                        tool_input=tool_input,
                        observation="",
                        is_finish=True,
                    )
                )
                return AgentResult(
                    goal=goal,
                    final_answer=str(answer),
                    steps=steps,
                    stop_reason=StopReason.FINISHED,
                    usage=usage,
                )

            # ── Execute tool asynchronously ─────────────────────────────────
            observation, duration_ms = await self._executor.aexecute(
                tool_name, tool_input
            )
            usage.tools_called += 1

            steps.append(
                AgentStep(
                    step_num=step_num,
                    thought=thought,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    observation=observation,
                    duration_ms=duration_ms,
                )
            )

        # ── Max steps reached ───────────────────────────────────────────────
        return AgentResult(
            goal=goal,
            final_answer=None,
            steps=steps,
            stop_reason=StopReason.MAX_STEPS,
            usage=usage,
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def registered_tools(self) -> list[str]:
        """Sorted list of all registered tool names."""
        return self._executor.names

    @property
    def system_prompt(self) -> str:
        """The system prompt sent to the LLM each step (read-only)."""
        return self._system_prompt


# ---------------------------------------------------------------------------
# Async-only variant (for FastAPI / async servers)
# ---------------------------------------------------------------------------


class AsyncAgentPipeline:
    """Async-only variant of :class:`AgentPipeline`.

    Exposes a single ``async run()`` method - suitable for FastAPI endpoints
    where a sync ``run()`` should not be in the public API.

    All constructor parameters are identical to :class:`AgentPipeline`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._inner = AgentPipeline(*args, **kwargs)

    async def run(
        self,
        goal: str,
        **kwargs: Any,
    ) -> AgentResult:
        """Async-only agent entrypoint. See :meth:`AgentPipeline.arun`."""
        return await self._inner.arun(goal, **kwargs)

    @property
    def registered_tools(self) -> list[str]:
        """Sorted list of all registered tool names."""
        return self._inner.registered_tools

    @property
    def system_prompt(self) -> str:
        """The system prompt used by the inner pipeline."""
        return self._inner.system_prompt
