"""AgentPipeline - autonomous ReAct agent with pluggable tools.

Implements the Reason+Act (ReAct) loop:

  1. The LLM receives a system prompt listing all registered tools and a
     growing conversation transcript (goal + previous steps).
  2. The LLM responds with a JSON object — either a single tool call:
     ``{"thought": "...", "tool_name": "...", "tool_input": {...}}``
     or a parallel batch:
     ``{"thought": "...", "tool_calls": [{"tool_name": "...", "tool_input": {...}}, ...]}``
  3. The tool(s) are executed; results (observations) are appended to the
     transcript.
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
        max_parallel_tools=4,
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
from concurrent.futures import ThreadPoolExecutor
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

# Built-in tool name for dynamic step extension
REQUEST_STEPS_TOOL = "request_more_steps"

# Type alias for one parsed tool call
_ToolCall = tuple[str, dict[str, Any]]

# Sentinel for "not provided by caller"
_UNSET = object()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_MEMO_PREVIEW_CHARS = 200

_SYSTEM_PROMPT_TEMPLATE = """\
You are an autonomous AI agent. You solve tasks step-by-step by calling the \
tools listed below. Think carefully before each action.

AVAILABLE TOOLS:
{tool_descriptions}

RESPONSE FORMAT
---------------
Single tool call (default):
{{"thought": "<your reasoning>", "tool_name": "<tool_name>", "tool_input": {{<key-value args>}}}}

Multiple independent tools in one step (parallel):
{{"thought": "<your reasoning>", "tool_calls": [{{"tool_name": "...", "tool_input": {{...}}}}, ...]}}

Output ONLY a single valid JSON object — no extra text, no markdown.

RULES:
1. Always include "thought" and either "tool_name"+"tool_input" OR "tool_calls".
2. To finish, call the finish tool:
   {{"thought": "I have the answer", "tool_name": "finish", "tool_input": {{"answer": "<full answer>"}}}}
3. If a tool returns an ERROR, adjust your approach and try a different strategy. Do NOT retry \
the exact same tool call with identical inputs.
4. Never invent data — only use facts from tool observations.
5. The "thought" field is your private reasoning — be concise.
6. Memory check: before calling any tool, scan previous observations. If you already called that \
tool with identical inputs and received a result, reuse that observation — do not call it again.
7. Early exit: if a tool error is clearly unrecoverable (service unavailable, data not found, \
missing dependency), call finish immediately with a partial answer that states what succeeded, \
what failed, and why — do not keep retrying.
8. Parallel execution: when multiple independent tool calls are needed, use the "tool_calls" \
array format so they run simultaneously — this is faster than sequential calls.
{extra_rules}"""


def _build_system_prompt(executor: ToolExecutor, extra_rules: str = "") -> str:
    extra = f"9. {extra_rules}" if extra_rules else ""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=executor.describe_all(),
        extra_rules=extra,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_key(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Stable deduplication key for a tool call."""
    return f"{tool_name}({json.dumps(tool_input, sort_keys=True)})"


def _build_transcript(
    goal: str,
    steps: list[AgentStep],
    seen_calls: dict[str, str] | None = None,
) -> str:
    """Build the rolling conversation transcript passed to the LLM each step."""
    lines: list[str] = [f"TASK: {goal}\n"]

    # Inject deduplication memo so the LLM avoids redundant tool calls
    if seen_calls:
        memo = ["PRIOR TOOL RESULTS (reuse these — do not repeat the same calls):"]
        for key, obs in seen_calls.items():
            preview = obs[:_MEMO_PREVIEW_CHARS] + ("..." if len(obs) > _MEMO_PREVIEW_CHARS else "")
            memo.append(f"  {key}  =>  {preview}")
        lines.append("\n".join(memo) + "\n")

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


def _parse_response(
    text: str,
) -> tuple[str | None, list[_ToolCall]]:
    """Parse the LLM text into ``(thought, list_of_tool_calls)``.

    Handles:

    - Single-call format: ``{"thought": "...", "tool_name": "...", "tool_input": {...}}``
    - Parallel-call format: ``{"thought": "...", "tool_calls": [...]}``
    - JSON inside markdown code fences
    - Malformed responses (falls back to ``finish`` with raw text as answer)
    """
    raw = text.strip()

    # Strip markdown code fences  ``` or ```json
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Extract the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        data = json.loads(raw)
        thought: str | None = data.get("thought") or None

        # ── Parallel format ──────────────────────────────────────────────
        tool_calls_raw = data.get("tool_calls")
        if isinstance(tool_calls_raw, list) and tool_calls_raw:
            calls: list[_ToolCall] = []
            for item in tool_calls_raw:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("tool_name", FINISH_TOOL)).strip()
                inp = item.get("tool_input", {})
                if not isinstance(inp, dict):
                    inp = {"value": str(inp)}
                calls.append((name, inp))
            if calls:
                return thought, calls

        # ── Single-call format ───────────────────────────────────────────
        tool_name = str(data.get("tool_name", FINISH_TOOL)).strip()
        tool_input = data.get("tool_input", {})
        if not isinstance(tool_input, dict):
            tool_input = {"value": str(tool_input)}
        return thought, [(tool_name, tool_input)]

    except (json.JSONDecodeError, ValueError):
        return None, [(FINISH_TOOL, {"answer": text})]


def _build_partial_answer(goal: str, steps: list[AgentStep], last_error: str) -> str:
    """Summarise what succeeded before the circuit breaker fired."""
    successes = [
        f"• {s.tool_name}: {s.observation[:200]}"
        for s in steps
        if not s.observation.startswith("ERROR") and not s.is_finish
    ]
    body = "\n".join(successes) if successes else "No tools completed successfully."
    return (
        f"The agent was unable to complete the full task due to repeated tool failures.\n\n"
        f"**Goal:** {goal}\n\n"
        f"**What succeeded:**\n{body}\n\n"
        f"**Last error:** {last_error}"
    )


def _coerce_response_format(
    answer: str,
    model_class: Any,
    kit: Any,
    prompt: Any,
) -> Any:
    """Parse *answer* into *model_class*.

    Attempt 1: direct JSON parse + Pydantic validation.
    Attempt 2: one LLM correction call to reformat the answer.
    Returns ``None`` on total failure.
    """
    # Attempt 1 — direct parse
    try:
        raw = answer.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        return model_class.model_validate(json.loads(raw))
    except Exception:
        pass

    # Attempt 2 — ask LLM to reformat
    try:
        from ractogateway._models.chat import ChatConfig

        msg = (
            f"Reformat the following text as a valid JSON object matching the "
            f"{model_class.__name__} schema. Return ONLY the JSON object.\n\nText:\n{answer}"
        )
        resp = kit.chat(ChatConfig(user_message=msg, prompt=prompt))
        raw = (resp.content or "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        return model_class.model_validate(json.loads(raw))
    except Exception:
        return None


async def _acoerce_response_format(
    answer: str,
    model_class: Any,
    kit: Any,
    prompt: Any,
) -> Any:
    """Async variant of :func:`_coerce_response_format`."""
    # Attempt 1 — direct parse
    try:
        raw = answer.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        return model_class.model_validate(json.loads(raw))
    except Exception:
        pass

    # Attempt 2 — ask LLM to reformat
    try:
        from ractogateway._models.chat import ChatConfig

        msg = (
            f"Reformat the following text as a valid JSON object matching the "
            f"{model_class.__name__} schema. Return ONLY the JSON object.\n\nText:\n{answer}"
        )
        resp = await kit.achat(ChatConfig(user_message=msg, prompt=prompt))
        raw = (resp.content or "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        return model_class.model_validate(json.loads(raw))
    except Exception:
        return None


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
        Optional ``RactoRAG`` instance — auto-registers a ``rag_search`` tool.
    sql_pipeline:
        Optional ``SQLAnalystPipeline`` — auto-registers a ``sql_query`` tool.
    enable_http:
        When ``True``, registers an ``http_get`` tool that fetches URLs via
        ``httpx`` (requires ``pip install ractogateway[pipelines-agent-http]``).
    agent_memory:
        Any dict-like or object with ``get``/``set`` methods.  Auto-registers
        ``memory_read`` and ``memory_write`` tools.
    max_steps:
        Hard cap on reasoning steps before the loop stops with
        :attr:`StopReason.MAX_STEPS`.
    max_consecutive_errors:
        Number of consecutive tool errors that trigger the circuit breaker
        (:attr:`StopReason.CIRCUIT_BREAK`).  Default ``3``.
    tool_retries:
        How many times to retry a failing tool before reporting the error to
        the LLM.  Default ``0`` (no retry).
    max_step_extension:
        Maximum additional steps the agent may request via
        ``request_more_steps``.  ``0`` disables the feature.
    max_parallel_tools:
        Maximum number of tools to run simultaneously when the LLM requests a
        parallel batch.  ``1`` forces sequential execution.  Default ``4``.
    system_prompt:
        Fully override the auto-generated system prompt (advanced).
    extra_rules:
        Append an extra numbered rule to the default system prompt.
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
        max_consecutive_errors: int = 3,
        tool_retries: int = 0,
        max_step_extension: int = 0,
        max_parallel_tools: int = 4,
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
        self._max_consecutive_errors = max_consecutive_errors
        self._max_step_extension = max_step_extension
        self._max_parallel_tools = max(1, max_parallel_tools)
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

        # 6. Dynamic step extension (intercepted in loop; executor never calls it)
        def request_more_steps(additional: int = 3, _reason: str = "") -> str:
            """Request additional reasoning steps when the task needs more work."""
            return f"Requested {additional} more steps."

        tool_map[REQUEST_STEPS_TOOL] = request_more_steps

        # 7. User-supplied tools (last, so they can shadow built-ins)
        for fn in tools or []:
            name: str = getattr(fn, "__tool_name__", None) or fn.__name__
            tool_map[name] = fn

        self._executor = ToolExecutor(tool_map, max_retries=tool_retries)

        # ── System prompt ─────────────────────────────────────────────────
        self._system_prompt = (
            system_prompt
            if system_prompt is not None
            else _build_system_prompt(self._executor, extra_rules)
        )

        # ── RactoPrompt for ChatConfig ─────────────────────────────────────
        from ractogateway.prompts.engine import RactoPrompt

        self._prompt = RactoPrompt(
            role="You are an autonomous AI agent.",
            aim=self._system_prompt,
            constraints=["Follow all instructions in the aim section exactly."],
            tone="Systematic, precise, and tool-focused.",
            output_format=(
                'Single: {"thought": "...", "tool_name": "...", "tool_input": {...}} '
                'OR parallel: {"thought": "...", "tool_calls": [{"tool_name": ..., "tool_input": ...}, ...]}'
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
        response_format: type | None = None,
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
        response_format:
            A Pydantic ``BaseModel`` subclass.  When provided, the agent's
            final answer is parsed and validated into an instance stored in
            :attr:`AgentResult.parsed_output`.

        Returns
        -------
        AgentResult
            Contains the final answer, all steps, stop reason, token usage,
            and (when *response_format* is set) the parsed structured output.
        """
        uid = self._user_id if user_id is _UNSET else user_id
        steps_cap = self._max_steps if max_steps is _UNSET else int(max_steps)

        if self._safe_mode:
            try:
                return self._run_loop(goal, steps_cap, uid, session_id, response_format)
            except Exception as exc:
                return AgentResult(
                    goal=goal,
                    stop_reason=StopReason.ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                )
        return self._run_loop(goal, steps_cap, uid, session_id, response_format)

    # ── Public async API ─────────────────────────────────────────────────────

    async def arun(
        self,
        goal: str,
        *,
        max_steps: Any = _UNSET,
        user_id: Any = _UNSET,
        session_id: str | None = None,
        response_format: type | None = None,
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
        response_format:
            A Pydantic ``BaseModel`` subclass for structured output parsing.
        """
        uid = self._user_id if user_id is _UNSET else user_id
        steps_cap = self._max_steps if max_steps is _UNSET else int(max_steps)

        if self._safe_mode:
            try:
                return await self._arun_loop(goal, steps_cap, uid, session_id, response_format)
            except Exception as exc:
                return AgentResult(
                    goal=goal,
                    stop_reason=StopReason.ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                )
        return await self._arun_loop(goal, steps_cap, uid, session_id, response_format)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _filter_calls(
        self,
        calls: list[_ToolCall],
        step_cap: int,
        steps: list[AgentStep],
        step_num: int,
        thought: str | None,
    ) -> tuple[list[_ToolCall], int]:
        """Filter finish/request_more_steps from *calls*; apply step extension.

        Returns ``(tool_calls, updated_step_cap)``.
        """
        tool_calls: list[_ToolCall] = []
        for call_name, call_inp in calls:
            if call_name == FINISH_TOOL:
                continue
            if call_name == REQUEST_STEPS_TOOL:
                additional = min(
                    int(call_inp.get("additional", 3)),
                    self._max_step_extension,
                )
                if additional > 0:
                    step_cap += additional
                steps.append(
                    AgentStep(
                        step_num=step_num,
                        thought=thought,
                        tool_name=REQUEST_STEPS_TOOL,
                        tool_input=call_inp,
                        observation=f"Granted {additional} extra steps. New cap: {step_cap}.",
                        duration_ms=0.0,
                    )
                )
            else:
                tool_calls.append((call_name, call_inp))
        return tool_calls, step_cap

    def _exec_sync(self, tool_calls: list[_ToolCall]) -> list[tuple[str, float]]:
        """Execute tool calls synchronously, in parallel when appropriate."""
        if len(tool_calls) == 1 or self._max_parallel_tools <= 1:
            return [self._executor.execute(name, inp) for name, inp in tool_calls]
        workers = min(len(tool_calls), self._max_parallel_tools)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._executor.execute, name, inp) for name, inp in tool_calls
            ]
            return [f.result() for f in futures]

    async def _exec_async(self, tool_calls: list[_ToolCall]) -> list[tuple[str, float]]:
        """Execute tool calls asynchronously, in parallel when appropriate."""
        if len(tool_calls) == 1 or self._max_parallel_tools <= 1:
            return [await self._executor.aexecute(name, inp) for name, inp in tool_calls]
        capped = tool_calls[: self._max_parallel_tools]
        overflow = tool_calls[self._max_parallel_tools :]
        results: list[tuple[str, float]] = list(
            await self._executor.aexecute_parallel(capped)
        )
        for name, inp in overflow:
            results.append(await self._executor.aexecute(name, inp))
        return results

    def _apply_results(
        self,
        tool_calls: list[_ToolCall],
        exec_results: list[tuple[str, float]],
        steps: list[AgentStep],
        seen_calls: dict[str, str],
        step_num: int,
        thought: str | None,
        consecutive_errors: int,
        goal: str,
        usage: AgentUsage,
    ) -> tuple[int, AgentResult | None]:
        """Record tool results into *steps*/*seen_calls*; return circuit-break result if triggered."""
        step_thought: str | None = thought
        for (name, inp), (obs, dur) in zip(tool_calls, exec_results, strict=False):
            seen_calls[_call_key(name, inp)] = obs
            steps.append(
                AgentStep(
                    step_num=step_num,
                    thought=step_thought,
                    tool_name=name,
                    tool_input=inp,
                    observation=obs,
                    duration_ms=dur,
                )
            )
            step_thought = None
            if obs.startswith("ERROR"):
                consecutive_errors += 1
                if consecutive_errors >= self._max_consecutive_errors:
                    partial = _build_partial_answer(goal, steps, obs)
                    return consecutive_errors, AgentResult(
                        goal=goal,
                        final_answer=partial,
                        steps=steps,
                        stop_reason=StopReason.CIRCUIT_BREAK,
                        error=f"Circuit breaker: {consecutive_errors} consecutive tool errors.",
                        usage=usage,
                    )
            else:
                consecutive_errors = 0
        return consecutive_errors, None

    # ── Sync execution loop ───────────────────────────────────────────────────

    def _run_loop(
        self,
        goal: str,
        max_steps: int,
        user_id: str,
        _session_id: str | None,
        response_format: type | None,
    ) -> AgentResult:
        from ractogateway._models.chat import ChatConfig

        self._check_rate_limit(user_id)

        usage = AgentUsage()
        steps: list[AgentStep] = []
        consecutive_errors = 0
        seen_calls: dict[str, str] = {}
        step_cap = max_steps
        step_num = 0

        while step_num < step_cap:
            step_num += 1

            transcript = _build_transcript(goal, steps, seen_calls)
            response = self._kit.chat(ChatConfig(user_message=transcript, prompt=self._prompt))

            if hasattr(response, "usage") and isinstance(response.usage, dict):
                raw: dict[str, Any] = response.usage
                in_tok: int = int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0)
                out_tok: int = int(raw.get("output_tokens") or raw.get("completion_tokens") or 0)
                usage.total_input_tokens += in_tok
                usage.total_output_tokens += out_tok
            usage.steps_taken = step_num

            thought, calls = _parse_response(response.content or "")

            # ── Finish (single finish call) ──────────────────────────────
            if len(calls) == 1 and calls[0][0] == FINISH_TOOL:
                tool_input = calls[0][1]
                answer = tool_input.get("answer", response.content or "")
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
                parsed = (
                    _coerce_response_format(str(answer), response_format, self._kit, self._prompt)
                    if response_format is not None
                    else None
                )
                return AgentResult(
                    goal=goal,
                    final_answer=str(answer),
                    steps=steps,
                    stop_reason=StopReason.FINISHED,
                    usage=usage,
                    parsed_output=parsed,
                )

            # ── Filter calls; intercept request_more_steps ───────────────
            tool_calls, step_cap = self._filter_calls(calls, step_cap, steps, step_num, thought)
            if not tool_calls:
                continue

            # ── Execute tools (parallel if multiple) ─────────────────────
            exec_results = self._exec_sync(tool_calls)
            usage.tools_called += len(exec_results)

            # ── Record results; check circuit breaker ─────────────────────
            consecutive_errors, cb_result = self._apply_results(
                tool_calls, exec_results, steps, seen_calls, step_num, thought,
                consecutive_errors, goal, usage,
            )
            if cb_result is not None:
                return cb_result

        # ── Max steps reached ────────────────────────────────────────────
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
        _session_id: str | None,
        response_format: type | None,
    ) -> AgentResult:
        from ractogateway._models.chat import ChatConfig

        self._check_rate_limit(user_id)

        usage = AgentUsage()
        steps: list[AgentStep] = []
        consecutive_errors = 0
        seen_calls: dict[str, str] = {}
        step_cap = max_steps
        step_num = 0

        while step_num < step_cap:
            step_num += 1

            transcript = _build_transcript(goal, steps, seen_calls)
            response = await self._kit.achat(
                ChatConfig(user_message=transcript, prompt=self._prompt)
            )

            if hasattr(response, "usage") and isinstance(response.usage, dict):
                raw: dict[str, Any] = response.usage
                in_tok: int = int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0)
                out_tok: int = int(raw.get("output_tokens") or raw.get("completion_tokens") or 0)
                usage.total_input_tokens += in_tok
                usage.total_output_tokens += out_tok
            usage.steps_taken = step_num

            thought, calls = _parse_response(response.content or "")

            # ── Finish ───────────────────────────────────────────────────
            if len(calls) == 1 and calls[0][0] == FINISH_TOOL:
                tool_input = calls[0][1]
                answer = tool_input.get("answer", response.content or "")
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
                parsed = (
                    await _acoerce_response_format(
                        str(answer), response_format, self._kit, self._prompt
                    )
                    if response_format is not None
                    else None
                )
                return AgentResult(
                    goal=goal,
                    final_answer=str(answer),
                    steps=steps,
                    stop_reason=StopReason.FINISHED,
                    usage=usage,
                    parsed_output=parsed,
                )

            # ── Filter calls; intercept request_more_steps ───────────────
            tool_calls, step_cap = self._filter_calls(calls, step_cap, steps, step_num, thought)
            if not tool_calls:
                continue

            # ── Execute tools (parallel via asyncio.gather) ───────────────
            exec_results = await self._exec_async(tool_calls)
            usage.tools_called += len(exec_results)

            # ── Record results; check circuit breaker ─────────────────────
            consecutive_errors, cb_result = self._apply_results(
                tool_calls, exec_results, steps, seen_calls, step_num, thought,
                consecutive_errors, goal, usage,
            )
            if cb_result is not None:
                return cb_result

        # ── Max steps reached ────────────────────────────────────────────
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
