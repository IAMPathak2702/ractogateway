"""ListClassifierPipeline — map a natural-language query to items from a list.

Two classes are exported:

- :class:`ListClassifierPipeline` — ``run()`` is **sync**, ``arun()`` is **async**.
- :class:`AsyncListClassifierPipeline` — ``run()`` is **async** only.

The pipeline takes a Python ``list[str]`` of candidate options and a user query,
then asks the configured LLM to pick the best matching option(s).  Internally it
builds a dynamic Python :class:`enum.Enum` from the options list and validates
every LLM response against it — hallucinated or paraphrased answers are
automatically caught, fuzzy-corrected if possible, or retried.

Provider / model selection
--------------------------
Pass any RactoGateway kit directly::

    from ractogateway.openai_developer_kit import Chat as OpenAIChat
    from ractogateway.anthropic_developer_kit import Chat as ClaudeChat
    from ractogateway.google_developer_kit import Chat as GeminiChat
    from ractogateway.ollama_developer_kit import Chat as OllamaChat
    from ractogateway.huggingface_developer_kit import Chat as HFChat

    pipeline = ListClassifierPipeline(kit=OpenAIChat(model="gpt-4o-mini"), ...)

Or use the convenience factory — no separate import needed::

    pipeline = ListClassifierPipeline.from_provider(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        options=["Billing", "Support", "Sales"],
    )

    # Supported providers: "openai", "anthropic", "google", "ollama", "huggingface"

Selection modes
---------------
- ``"single"``   — LLM must return exactly one option (default).
- ``"multiple"`` — LLM may return one or more options from the list.

Output formats
--------------
- ``"pydantic"`` — returns a :class:`ClassifierResult` (default).
- ``"string"``   — returns a comma-joined string (e.g. ``"Billing, Account"``).
- ``"dict"``     — returns ``{"selected": [...], "confidences": [...], ...}``.

Production features
-------------------
- **Dynamic Enum validation** — options → Python ``Enum`` per-call; invalid
  LLM responses auto-retried (up to ``max_retries``, default 2).
- **Fuzzy fallback** — stdlib ``difflib`` fuzzy-matches near-misses before
  consuming a retry (``fuzzy_fallback=True``, default).
- **Option descriptions** — per-option natural-language descriptions help the
  LLM distinguish similar-sounding categories.
- **Score-all mode** — ``score_all=True`` asks the LLM for a confidence score
  for every option, not just the selected ones; stored in
  ``result.all_scores``.
- **Uncertain label** — ``uncertain_label="Other"`` injects a catch-all option
  so the LLM has somewhere to fall when nothing matches.
- **Batch classification** — ``batch_run(queries)`` / ``abatch_run(queries)``
  run multiple queries; async version runs them concurrently.
- **Runtime option management** — ``add_option()``, ``remove_option()``,
  ``set_options()``, ``get_options()`` mutate the pipeline's default list.
- **Safe mode** — exceptions → ``result.error`` field instead of raising.
- **Telemetry** — ``tracer=RactoTracer(...)`` (OTEL) + ``metrics=GatewayMetricsMiddleware(...)``.
- **Rate limiting** — duck-typed ``rate_limiter=``.
- **Conversation memory** — duck-typed ``memory=`` + per-call ``session_id``.
- **Case-insensitive matching** (default).
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from ractogateway._models.chat import ChatConfig
from ractogateway.adapters.base import LLMResponse
from ractogateway.pipelines.list_classifier._models import (
    AuditEntry,
    ClassifierRateLimitExceededError,
    ClassifierResult,
    ClassifierUsage,
)
from ractogateway.prompts.engine import RactoPrompt

# ---------------------------------------------------------------------------
# Sentinel — distinguishes "caller passed None" from "not provided"
# ---------------------------------------------------------------------------

_UNSET: Any = object()

# ---------------------------------------------------------------------------
# Provider → kit factory
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, tuple[str, str]] = {
    "openai":       ("ractogateway.openai_developer_kit",       "OpenAIDeveloperKit"),
    "anthropic":    ("ractogateway.anthropic_developer_kit",    "AnthropicDeveloperKit"),
    "google":       ("ractogateway.google_developer_kit",       "GoogleDeveloperKit"),
    "ollama":       ("ractogateway.ollama_developer_kit",       "OllamaDeveloperKit"),
    "huggingface":  ("ractogateway.huggingface_developer_kit",  "HuggingFaceDeveloperKit"),
}


def _build_kit(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Lazily import and instantiate a provider kit by name.

    Parameters
    ----------
    provider:
        One of ``"openai"``, ``"anthropic"``, ``"google"``, ``"ollama"``,
        ``"huggingface"``.
    model:
        Model identifier string for the chosen provider.
    api_key:
        Provider API key (falls back to the relevant environment variable).
    base_url:
        Custom base URL — useful for Ollama or OpenAI-compatible proxies.
    """
    provider_lower = provider.lower()
    if provider_lower not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Supported: {sorted(_PROVIDER_MAP)}"
        )

    module_path, class_name = _PROVIDER_MAP[provider_lower]
    try:
        from importlib import import_module  # noqa: PLC0415
        mod = import_module(module_path)
        kit_cls = getattr(mod, class_name)
    except ImportError as exc:
        raise ImportError(
            f"Provider '{provider}' requires an extra dependency. "
            f"Install it with:  pip install ractogateway[{provider_lower}]"
        ) from exc

    kwargs: dict[str, Any] = {"model": model}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    return kit_cls(**kwargs)


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

_DEFAULT_CLASSIFIER_PROMPT = RactoPrompt(
    role=(
        "You are a precise semantic classification engine. "
        "Your sole job is to match a user query to the best option(s) "
        "from a numbered list provided in each message."
    ),
    aim=(
        "Analyse the semantic intent of the user query and select the option(s) "
        "that best match that intent.  Respond with ONLY a single valid JSON "
        "object — no prose, no markdown code fences, no text outside the JSON."
    ),
    constraints=[
        "SELECTION: Only select options that appear verbatim in the provided "
        "numbered list.  Never invent, paraphrase, or abbreviate option names.",
        "FORMAT: Output ONLY a valid JSON object.  No ```json fences, "
        "no leading/trailing text, no comments inside the JSON.",
        "EXACTNESS: Copy option strings exactly — same capitalisation, "
        "same spacing, same punctuation as shown in the list.",
        "CONFIDENCE: Confidence values must be floats in [0.0, 1.0]. "
        "List multiple selections in descending confidence order (best match first).",
        "NO_MATCH: If no option matches well, still pick the single closest option "
        "and assign it a low confidence (< 0.3) rather than returning empty.",
        "SINGLE_MODE: In single-selection mode 'selected' must be a plain string, "
        "not a list.",
        "MULTI_MODE: In multiple-selection mode 'selected' must be a JSON array "
        "of strings.",
    ],
    tone="precise, structured, and terse",
    output_format=(
        "A single valid JSON object whose schema is shown in the user message. "
        "Nothing else."
    ),
    anti_hallucination=True,
)

# ---------------------------------------------------------------------------
# JSON extraction + normalisation helpers
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> str:
    """Strip markdown fences and return the innermost JSON object string."""
    text = text.strip()
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _norm(s: str, case_sensitive: bool) -> str:
    return s if case_sensitive else s.lower().strip()


def _match_exact(raw: str, options: list[str], case_sensitive: bool) -> str | None:
    """Exact (case-aware) match."""
    raw_n = _norm(raw, case_sensitive)
    for opt in options:
        if _norm(opt, case_sensitive) == raw_n:
            return opt
    return None


def _match_fuzzy(raw: str, options: list[str], case_sensitive: bool) -> str | None:
    """stdlib difflib fuzzy match — returns best match above 0.6 cutoff."""
    norm_options = {_norm(o, case_sensitive): o for o in options}
    candidates = list(norm_options.keys())
    matches = difflib.get_close_matches(
        _norm(raw, case_sensitive), candidates, n=1, cutoff=0.6
    )
    return norm_options[matches[0]] if matches else None


def _build_options_enum(options: list[str]) -> type[Enum]:
    """Build a dynamic :class:`enum.Enum` from an options list."""
    return Enum("OptionsEnum", {opt: opt for opt in options})  # type: ignore[return-value]


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Pipeline-level cache helpers
#
# We reuse the existing ExactMatchCache / SemanticCache infrastructure by
# serialising ClassifierResult → JSON → LLMResponse.content for storage, and
# deserialising on a cache hit.  This gives us LRU, TTL, and thread-safety for
# free without adding new dependencies.
# ---------------------------------------------------------------------------


def _exact_cache_key(
    user_query: str,
    options: list[str],
    selection_mode: str,
    include_confidence: bool,
    include_reasoning: bool,
    score_all: bool,
    temperature: float,
    max_tokens: int,
    confidence_threshold: float | None,
) -> tuple[str, str, str, float, int]:
    """Build the 5-tuple used as an ExactMatchCache lookup key.

    Maps classifier-specific context onto the cache's
    ``(user_message, system_prompt, model, temperature, max_tokens)`` API.
    """
    # Encode the options list order-sensitively (same list = same key)
    opts_sig = hashlib.md5("|".join(options).encode(), usedforsecurity=False).hexdigest()
    system_prompt = (
        f"clf:{selection_mode}:"
        f"conf={include_confidence}:reason={include_reasoning}:"
        f"score_all={score_all}:thresh={confidence_threshold}:"
        f"opts={opts_sig}"
    )
    return user_query, system_prompt, "list_classifier", temperature, max_tokens


def _result_to_llm_response(result: ClassifierResult) -> LLMResponse:
    """Serialise *result* into a fake :class:`LLMResponse` for cache storage."""
    return LLMResponse(content=result.model_dump_json())


def _llm_response_to_result(response: LLMResponse) -> ClassifierResult | None:
    """Deserialise a cached :class:`LLMResponse` back to a :class:`ClassifierResult`."""
    try:
        return ClassifierResult.model_validate_json(response.content or "")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# User-turn message builder
# ---------------------------------------------------------------------------


def _build_classifier_message(  # noqa: PLR0913
    user_query: str,
    options: list[str],
    selection_mode: str,
    include_confidence: bool,
    include_reasoning: bool,
    score_all: bool,
    option_descriptions: dict[str, str] | None,
    memory_ctx: str = "",
    prior_response: str = "",
    error_msg: str = "",
) -> str:
    """Build the full user-turn message for the classifier LLM call."""
    # Numbered options — inject descriptions when available
    option_lines: list[str] = []
    for i, opt in enumerate(options):
        line = f'{i + 1}. "{opt}"'
        desc = (option_descriptions or {}).get(opt)
        if desc:
            line += f"  —  {desc}"
        option_lines.append(line)
    numbered = "\n".join(option_lines)

    if selection_mode == "single":
        mode_line = "Select EXACTLY ONE option (the best semantic match)."
        selected_schema = f'"selected": "<one of the {len(options)} option strings above>"'
        confidence_schema = '"confidence": <float 0.0–1.0>'
    else:
        mode_line = (
            "Select ONE OR MORE options (all that semantically match the query). "
            "Order selections by descending relevance."
        )
        selected_schema = '"selected": ["<option>", ...]'
        confidence_schema = '"confidences": [<float>, ...]  // same length as selected, descending'

    schema_lines = [selected_schema]
    if include_confidence:
        schema_lines.append(confidence_schema)
    if score_all:
        schema_lines.append(
            '"all_scores": {"<option>": <float>, ...}  // score for EVERY option'
        )
    if include_reasoning:
        schema_lines.append('"reasoning": "<one-sentence explanation>"')

    # Concrete example using first real option
    example_val = f'"{options[0]}"' if selection_mode == "single" else f'["{options[0]}"]'
    example_lines = [f'"selected": {example_val}']
    if include_confidence:
        if selection_mode == "single":
            example_lines.append('"confidence": 0.95')
        else:
            example_lines.append('"confidences": [0.95]')
    if score_all:
        scores_ex = ", ".join(
            f'"{o}": {round(0.9 - i * 0.15, 2)}'
            for i, o in enumerate(options[:4])
        )
        example_lines.append(f'"all_scores": {{{scores_ex}}}')
    if include_reasoning:
        example_lines.append('"reasoning": "The query closely matches this option."')

    schema_str  = "{\n  " + ",\n  ".join(schema_lines) + "\n}"
    example_str = "{\n  " + ",\n  ".join(example_lines) + "\n}"

    retry_block = ""
    if prior_response and error_msg:
        retry_block = (
            f"\n\n---\n"
            f"**Your previous response was INVALID — do not repeat it:**\n"
            f"```\n{prior_response[:600]}\n```\n"
            f"**Error:** {error_msg}\n\n"
            f"Produce only the corrected JSON object.\n---"
        )

    return (
        f"{memory_ctx}"
        f"## Available Options\n\n"
        f"{numbered}\n\n"
        f"## Task\n\n"
        f"{mode_line}\n\n"
        f"## User Query\n\n"
        f"{user_query}\n\n"
        f"## Required JSON Schema\n\n"
        f"{schema_str}\n\n"
        f"## Example Response\n\n"
        f"{example_str}"
        f"{retry_block}"
    )


# ---------------------------------------------------------------------------
# Response parser / Enum validator
# ---------------------------------------------------------------------------


def _parse_and_validate(  # noqa: PLR0912, PLR0913
    raw_content: str,
    options: list[str],
    options_enum: type[Enum],
    selection_mode: str,
    include_confidence: bool,
    include_reasoning: bool,
    score_all: bool,
    case_sensitive: bool,
    confidence_threshold: float | None,
    fuzzy_fallback: bool,
) -> tuple[list[str], list[float] | None, dict[str, float] | None, str | None, bool]:
    """Parse and Enum-validate the LLM JSON response.

    Returns
    -------
    selected, confidences, all_scores, reasoning, fuzzy_corrected
    """
    json_text = _extract_json(raw_content)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON: {exc}. "
            f"Received: {json_text[:300]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object, got {type(data).__name__}."
        )

    # ── selected ─────────────────────────────────────────────────────────
    raw_selected = data.get("selected")
    if raw_selected is None:
        raise ValueError(
            f"LLM response missing 'selected' field. Keys found: {list(data.keys())}"
        )

    raw_list: list[str]
    if isinstance(raw_selected, str):
        raw_list = [raw_selected]
    elif isinstance(raw_selected, list):
        raw_list = [str(x) for x in raw_selected]
    else:
        raise ValueError(
            f"'selected' must be string or list, got {type(raw_selected).__name__}."
        )

    if not raw_list:
        raise ValueError("'selected' is empty — LLM must choose at least one option.")

    if selection_mode == "single":
        raw_list = raw_list[:1]

    # ── Enum validation with optional fuzzy fallback ──────────────────────
    validated: list[str] = []
    fuzzy_corrected = False

    for raw in raw_list:
        matched = _match_exact(raw, options, case_sensitive)

        if matched is None and fuzzy_fallback:
            matched = _match_fuzzy(raw, options, case_sensitive)
            if matched is not None:
                fuzzy_corrected = True

        if matched is None:
            close = difflib.get_close_matches(raw, options, n=3, cutoff=0.4)
            hint = f"  Closest: {close}" if close else ""
            raise ValueError(
                f"LLM returned {raw!r} which does not match any option.{hint} "
                f"Valid options: {options}"
            )

        # Confirm against the Enum (belt-and-suspenders)
        try:
            options_enum(matched)
        except ValueError as exc:
            raise ValueError(
                f"Option {matched!r} not in options Enum."
            ) from exc

        if matched not in validated:
            validated.append(matched)

    # ── confidences ───────────────────────────────────────────────────────
    confidences: list[float] | None = None
    if include_confidence:
        if selection_mode == "single":
            raw_conf = data.get("confidence")
            c = float(raw_conf) if raw_conf is not None else 1.0
            confidences = [max(0.0, min(1.0, c))]
        else:
            raw_confs = data.get("confidences")
            if isinstance(raw_confs, list) and raw_confs:
                confidences = [
                    max(0.0, min(1.0, float(c)))
                    for c in raw_confs[: len(validated)]
                ]
                while len(confidences) < len(validated):
                    confidences.append(1.0)
            else:
                confidences = [1.0] * len(validated)

        if confidence_threshold is not None and confidences:
            paired = list(zip(validated, confidences))
            filtered = [(o, c) for o, c in paired if c >= confidence_threshold]
            if filtered:
                validated = [p[0] for p in filtered]
                confidences = [p[1] for p in filtered]
            else:
                best = max(zip(validated, confidences), key=lambda x: x[1])
                validated, confidences = [best[0]], [best[1]]

    # ── all_scores ────────────────────────────────────────────────────────
    all_scores: dict[str, float] | None = None
    if score_all:
        raw_scores = data.get("all_scores") or {}
        if isinstance(raw_scores, dict):
            all_scores = {}
            for opt in options:
                # Try exact key first, then case-insensitive fallback
                score_raw = raw_scores.get(opt)
                if score_raw is None:
                    score_raw = next(
                        (v for k, v in raw_scores.items()
                         if _norm(k, False) == _norm(opt, False)),
                        None,
                    )
                all_scores[opt] = max(0.0, min(1.0, float(score_raw))) if score_raw is not None else 0.0

    # ── reasoning ─────────────────────────────────────────────────────────
    reasoning: str | None = None
    if include_reasoning:
        reasoning = str(data.get("reasoning", "")).strip() or None

    return validated, confidences, all_scores, reasoning, fuzzy_corrected


# ---------------------------------------------------------------------------
# ListClassifierPipeline
# ---------------------------------------------------------------------------


class ListClassifierPipeline:
    """Map a natural-language query to one or more items from a candidate list.

    Supports every RactoGateway provider via the ``kit`` parameter or the
    ``from_provider()`` class factory.  Internally builds a dynamic Python
    :class:`enum.Enum` from the options list and validates every LLM response
    against it — hallucinations and paraphrased answers are caught, fuzzy-
    corrected if close enough, and retried otherwise.

    Two variants
    ------------
    - :class:`ListClassifierPipeline` — ``run()`` sync, ``arun()`` async.
    - :class:`AsyncListClassifierPipeline` — ``run()`` is async only.

    Parameters
    ----------
    kit:
        Any RactoGateway developer kit (OpenAI, Anthropic, Google, Ollama,
        HuggingFace).  Must expose ``.chat(ChatConfig)`` and
        ``.achat(ChatConfig)`` methods.  Use ``from_provider()`` instead of
        constructing kits manually when you only need provider + model.
    options:
        Default candidate strings.  Can be overridden per-call.
        Must be non-empty and duplicate-free when provided.
    selection_mode:
        ``"single"`` (default) — exactly one option.
        ``"multiple"`` — one or more options.
        Overridable per-call.
    output_format:
        ``"pydantic"`` (default) — :class:`ClassifierResult`.
        ``"string"`` — comma-joined string.
        ``"dict"`` — plain ``dict``.
        Overridable per-call.
    prompt:
        Custom :class:`~ractogateway.prompts.engine.RactoPrompt` to replace
        the built-in system prompt.
    temperature:
        LLM temperature.  Default ``0.0`` for deterministic output.
    max_tokens:
        Response token budget.  Default ``512``.
    max_retries:
        Retry attempts when LLM returns invalid JSON / unknown option.
        Default ``2``.
    include_confidence:
        Ask LLM for per-selection confidence scores [0.0–1.0].  Default ``True``.
    include_reasoning:
        Ask LLM for a one-sentence explanation.  Default ``False``.
    score_all:
        Ask LLM for a score for *every* option (not just selected ones).
        Stored in ``result.all_scores``.  Default ``False``.
    option_descriptions:
        ``{option: description}`` — shown inline next to each option in the
        prompt to help the LLM distinguish similar categories.
    fuzzy_fallback:
        Use stdlib ``difflib`` to correct near-miss LLM responses before
        consuming a retry.  Default ``True``.
    uncertain_label:
        When set, this string is appended as an extra option that the LLM can
        pick when nothing matches (e.g. ``"Other / None of the above"``).
        ``result.uncertain`` is ``True`` when this label is selected.
    confidence_threshold:
        Drop selections below this score.  Keeps highest-confidence match as
        fallback.  Default ``None`` (no filtering).
    case_sensitive:
        Whether option matching is case-sensitive.  Default ``False``.
    safe_mode:
        Return ``ClassifierResult(error=...)`` instead of raising.
        Default ``False``.
    tracer:
        Optional :class:`~ractogateway.telemetry.RactoTracer`.
    metrics:
        Optional :class:`~ractogateway.telemetry.GatewayMetricsMiddleware`.
    rate_limiter:
        Duck-typed — ``check_and_consume(user_id, tokens) -> bool`` +
        ``get_remaining(user_id) -> int``.
    memory:
        Duck-typed — ``get_history(session_id) -> list[dict]`` +
        ``append(session_id, role, content)``.
    user_id:
        Default user ID for rate limiting.  Overridable per-call.

    Example
    -------
    ::

        # Via kit directly
        from ractogateway.openai_developer_kit import Chat
        from ractogateway.pipelines import ListClassifierPipeline

        pipeline = ListClassifierPipeline(
            kit=Chat(model="gpt-4o-mini"),
            options=["Billing", "Technical Support", "Sales"],
            include_confidence=True,
            include_reasoning=True,
        )
        result = pipeline.run("My invoice is wrong")
        print(result.first, result.top_confidence)

        # Via from_provider() — no manual kit import needed
        pipeline = ListClassifierPipeline.from_provider(
            "anthropic", "claude-haiku-4-5-20251001",
            options=["Billing", "Technical Support", "Sales"],
        )
    """

    def __init__(  # noqa: PLR0913
        self,
        kit: Any,
        *,
        options: list[str] | None = None,
        selection_mode: Literal["single", "multiple"] = "single",
        output_format: Literal["string", "dict", "pydantic"] = "pydantic",
        prompt: RactoPrompt | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 2,
        include_confidence: bool = True,
        include_reasoning: bool = False,
        score_all: bool = False,
        option_descriptions: dict[str, str] | None = None,
        fuzzy_fallback: bool = True,
        uncertain_label: str | None = None,
        confidence_threshold: float | None = None,
        case_sensitive: bool = False,
        safe_mode: bool = False,
        # ── Caching ────────────────────────────────────────────────────────
        exact_cache: Any | None = None,
        semantic_cache: Any | None = None,
        # ── Audit logging ───────────────────────────────────────────────────
        audit_logger: Any | None = None,
        # ── Observability ───────────────────────────────────────────────────
        tracer: Any | None = None,
        metrics: Any | None = None,
        # ── Rate limiting + memory + user ───────────────────────────────────
        rate_limiter: Any | None = None,
        memory: Any | None = None,
        user_id: str | None = None,
    ) -> None:
        if options is not None:
            if len(options) == 0:
                raise ValueError("options list cannot be empty.")
            if len(set(options)) != len(options):
                raise ValueError("options list contains duplicate entries.")

        _valid_modes = {"single", "multiple"}
        if selection_mode not in _valid_modes:
            raise ValueError(
                f"selection_mode must be one of {sorted(_valid_modes)}, "
                f"got {selection_mode!r}"
            )

        _valid_formats = {"string", "dict", "pydantic"}
        if output_format not in _valid_formats:
            raise ValueError(
                f"output_format must be one of {sorted(_valid_formats)}, "
                f"got {output_format!r}"
            )

        if confidence_threshold is not None and not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}."
            )

        self._kit = kit
        self._options: list[str] | None = list(options) if options is not None else None
        self._lock = threading.Lock()          # guards _options mutations
        self._selection_mode = selection_mode
        self._output_format = output_format
        self._prompt = prompt or _DEFAULT_CLASSIFIER_PROMPT
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._include_confidence = include_confidence
        self._include_reasoning = include_reasoning
        self._score_all = score_all
        self._option_descriptions = option_descriptions
        self._fuzzy_fallback = fuzzy_fallback
        self._uncertain_label = uncertain_label
        self._confidence_threshold = confidence_threshold
        self._case_sensitive = case_sensitive
        self._safe_mode = safe_mode
        self._exact_cache = exact_cache
        self._semantic_cache = semantic_cache
        self._audit_logger = audit_logger
        self._tracer = tracer
        self._metrics = metrics
        self._rate_limiter = rate_limiter
        self._memory = memory
        self._user_id = user_id

    # ------------------------------------------------------------------
    # Class-level factories
    # ------------------------------------------------------------------

    @classmethod
    def from_provider(  # noqa: PLR0913
        cls,
        provider: str,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        options: list[str] | None = None,
        **kwargs: Any,
    ) -> "ListClassifierPipeline":
        """Create a pipeline by specifying provider + model — no kit import needed.

        Parameters
        ----------
        provider:
            One of ``"openai"``, ``"anthropic"``, ``"google"``, ``"ollama"``,
            ``"huggingface"``.
        model:
            Model identifier string, e.g.:

            - OpenAI:      ``"gpt-4o-mini"``, ``"gpt-4o"``
            - Anthropic:   ``"claude-haiku-4-5-20251001"``, ``"claude-sonnet-4-6"``
            - Google:      ``"gemini-2.0-flash"``, ``"gemini-1.5-pro"``
            - Ollama:      ``"llama3.2"``, ``"mistral"``
            - HuggingFace: ``"meta-llama/Llama-3.2-3B-Instruct"``

        api_key:
            Provider API key.  Falls back to the standard env var for each
            provider (e.g. ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).
        base_url:
            Custom endpoint — used for Ollama (``http://localhost:11434``) or
            OpenAI-compatible proxies.
        options:
            Default candidate options list.
        **kwargs:
            Any other :class:`ListClassifierPipeline` constructor parameters
            (``selection_mode``, ``include_confidence``, ``safe_mode``, etc.).

        Returns
        -------
        ListClassifierPipeline

        Example
        -------
        ::

            pipeline = ListClassifierPipeline.from_provider(
                "anthropic",
                "claude-haiku-4-5-20251001",
                options=["Billing", "Support", "Sales"],
                include_reasoning=True,
                safe_mode=True,
            )
        """
        kit = _build_kit(provider, model, api_key=api_key, base_url=base_url)
        return cls(kit=kit, options=options, **kwargs)

    # ------------------------------------------------------------------
    # Class-level utility
    # ------------------------------------------------------------------

    @staticmethod
    def make_enum(options: list[str], name: str = "OptionsEnum") -> type[Enum]:
        """Build a standalone dynamic :class:`enum.Enum` from an options list.

        Useful when you want enum-typed values outside the pipeline.

        Parameters
        ----------
        options:
            List of option strings.
        name:
            Enum class name.  Default ``"OptionsEnum"``.

        Returns
        -------
        type[Enum]

        Example
        -------
        ::

            E = ListClassifierPipeline.make_enum(["Red", "Green", "Blue"])
            E["Red"].value   # "Red"
        """
        return _build_options_enum(options)

    # ------------------------------------------------------------------
    # Runtime option management (thread-safe)
    # ------------------------------------------------------------------

    def get_options(self) -> list[str] | None:
        """Return the pipeline-level options list, or ``None`` if not set."""
        with self._lock:
            return list(self._options) if self._options is not None else None

    def set_options(self, options: list[str]) -> None:
        """Replace the entire pipeline-level options list.

        Thread-safe — safe to call while the pipeline is in use.

        Parameters
        ----------
        options:
            New options list.  Must be non-empty and duplicate-free.
        """
        if len(options) == 0:
            raise ValueError("options list cannot be empty.")
        if len(set(options)) != len(options):
            raise ValueError("options list contains duplicate entries.")
        with self._lock:
            self._options = list(options)

    def add_option(self, option: str, description: str | None = None) -> None:
        """Append a new option to the pipeline-level list.

        Parameters
        ----------
        option:
            The option string to add.
        description:
            Optional inline description for the option.
        """
        with self._lock:
            if self._options is None:
                self._options = []
            if option in self._options:
                raise ValueError(f"Option {option!r} already exists in the list.")
            self._options.append(option)
            if description is not None:
                if self._option_descriptions is None:
                    self._option_descriptions = {}
                self._option_descriptions[option] = description

    def remove_option(self, option: str) -> None:
        """Remove an option from the pipeline-level list.

        Parameters
        ----------
        option:
            The option string to remove.  Raises ``ValueError`` if not found.
        """
        with self._lock:
            if self._options is None or option not in self._options:
                raise ValueError(f"Option {option!r} not found in the list.")
            self._options.remove(option)
            if self._option_descriptions:
                self._option_descriptions.pop(option, None)

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def run(  # noqa: PLR0913
        self,
        user_query: str,
        *,
        options: list[str] | None = _UNSET,
        selection_mode: Literal["single", "multiple"] | None = None,
        output_format: Literal["string", "dict", "pydantic"] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        confidence_threshold: float | None = _UNSET,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> ClassifierResult | str | dict[str, Any]:
        """Classify *user_query* synchronously.

        Parameters
        ----------
        user_query:
            Natural-language query to classify.
        options:
            Per-call override for the candidate list.  Omit to use the
            pipeline-level list.  Pass ``[]`` to get a ``ValueError``.
        selection_mode:
            Per-call override — ``"single"`` or ``"multiple"``.
        output_format:
            Per-call override — ``"pydantic"``, ``"string"``, or ``"dict"``.
        temperature / max_tokens:
            Per-call LLM setting overrides.
        confidence_threshold:
            Per-call override.  Pass ``None`` explicitly to disable filtering
            for this call even if a pipeline-level threshold is set.
        session_id:
            Conversation session ID for memory retrieval/storage.
        user_id:
            Per-call user ID for rate limiting and audit.

        Returns
        -------
        ClassifierResult | str | dict
            Type depends on *output_format*.
        """
        cfg = self._resolve_config(
            options=options, selection_mode=selection_mode,
            output_format=output_format, temperature=temperature,
            max_tokens=max_tokens, confidence_threshold=confidence_threshold,
            session_id=session_id, user_id=user_id,
        )
        ts = _utc_now()
        t0 = time.perf_counter()

        # ── Cache lookup (bypasses LLM + rate limit on hit) ──────────────
        cached = self._lookup_exact_cache(user_query, cfg)
        if cached is None:
            cached = self._lookup_semantic_cache(user_query)
        if cached is not None:
            latency_ms = (time.perf_counter() - t0) * 1000
            self._audit_log(cached, session_id=cfg["session_id"],
                            user_id=cfg["user_id"], latency_ms=latency_ms, timestamp=ts)
            return self._format_output(cached, cfg["output_format"])

        # ── Live LLM call ─────────────────────────────────────────────────
        if self._safe_mode:
            try:
                result = self._execute(user_query, cfg)
            except Exception as exc:  # noqa: BLE001
                result = ClassifierResult(
                    user_query=user_query,
                    options_provided=cfg["options"],
                    error=str(exc),
                )
        else:
            result = self._execute(user_query, cfg)

        latency_ms = (time.perf_counter() - t0) * 1000
        if not result.error:
            self._store_exact_cache(user_query, cfg, result)
            self._store_semantic_cache(user_query, result)
        self._audit_log(result, session_id=cfg["session_id"],
                        user_id=cfg["user_id"], latency_ms=latency_ms, timestamp=ts)
        return self._format_output(result, cfg["output_format"])

    def batch_run(  # noqa: PLR0913
        self,
        queries: list[str],
        *,
        options: list[str] | None = _UNSET,
        selection_mode: Literal["single", "multiple"] | None = None,
        output_format: Literal["string", "dict", "pydantic"] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        confidence_threshold: float | None = _UNSET,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[ClassifierResult | str | dict[str, Any]]:
        """Classify multiple queries synchronously, one after another.

        Shares all per-call overrides across every query in the batch.
        Use :meth:`abatch_run` to run them concurrently in async contexts.

        Parameters
        ----------
        queries:
            List of natural-language queries to classify.

        Returns
        -------
        list
            One result per query, in the same order.
        """
        shared: dict[str, Any] = dict(
            options=options, selection_mode=selection_mode,
            output_format=output_format, temperature=temperature,
            max_tokens=max_tokens, confidence_threshold=confidence_threshold,
            session_id=session_id, user_id=user_id,
        )
        return [self.run(q, **shared) for q in queries]

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def arun(  # noqa: PLR0913
        self,
        user_query: str,
        *,
        options: list[str] | None = _UNSET,
        selection_mode: Literal["single", "multiple"] | None = None,
        output_format: Literal["string", "dict", "pydantic"] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        confidence_threshold: float | None = _UNSET,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> ClassifierResult | str | dict[str, Any]:
        """Async variant of :meth:`run` — identical parameters."""
        cfg = self._resolve_config(
            options=options, selection_mode=selection_mode,
            output_format=output_format, temperature=temperature,
            max_tokens=max_tokens, confidence_threshold=confidence_threshold,
            session_id=session_id, user_id=user_id,
        )
        ts = _utc_now()
        t0 = time.perf_counter()

        # ── Cache lookup (bypasses LLM + rate limit on hit) ──────────────
        cached = self._lookup_exact_cache(user_query, cfg)
        if cached is None:
            cached = self._lookup_semantic_cache(user_query)
        if cached is not None:
            latency_ms = (time.perf_counter() - t0) * 1000
            self._audit_log(cached, session_id=cfg["session_id"],
                            user_id=cfg["user_id"], latency_ms=latency_ms, timestamp=ts)
            return self._format_output(cached, cfg["output_format"])

        # ── Live LLM call ─────────────────────────────────────────────────
        if self._safe_mode:
            try:
                result = await self._aexecute(user_query, cfg)
            except Exception as exc:  # noqa: BLE001
                result = ClassifierResult(
                    user_query=user_query,
                    options_provided=cfg["options"],
                    error=str(exc),
                )
        else:
            result = await self._aexecute(user_query, cfg)

        latency_ms = (time.perf_counter() - t0) * 1000
        if not result.error:
            self._store_exact_cache(user_query, cfg, result)
            self._store_semantic_cache(user_query, result)
        self._audit_log(result, session_id=cfg["session_id"],
                        user_id=cfg["user_id"], latency_ms=latency_ms, timestamp=ts)
        return self._format_output(result, cfg["output_format"])

    async def abatch_run(  # noqa: PLR0913
        self,
        queries: list[str],
        *,
        options: list[str] | None = _UNSET,
        selection_mode: Literal["single", "multiple"] | None = None,
        output_format: Literal["string", "dict", "pydantic"] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        confidence_threshold: float | None = _UNSET,
        session_id: str | None = None,
        user_id: str | None = None,
        max_concurrency: int | None = None,
    ) -> list[ClassifierResult | str | dict[str, Any]]:
        """Classify multiple queries concurrently with ``asyncio.gather``.

        Parameters
        ----------
        queries:
            List of natural-language queries.
        max_concurrency:
            Cap the number of simultaneous LLM calls.  ``None`` (default)
            runs all queries in parallel.  Set to e.g. ``5`` to avoid
            rate-limit errors on large batches.

        Returns
        -------
        list
            Results in the same order as *queries*.
        """
        shared: dict[str, Any] = dict(
            options=options, selection_mode=selection_mode,
            output_format=output_format, temperature=temperature,
            max_tokens=max_tokens, confidence_threshold=confidence_threshold,
            session_id=session_id, user_id=user_id,
        )

        if max_concurrency is None:
            return list(await asyncio.gather(*(self.arun(q, **shared) for q in queries)))

        # Semaphore-limited concurrency
        sem = asyncio.Semaphore(max_concurrency)

        async def _guarded(q: str) -> ClassifierResult | str | dict[str, Any]:
            async with sem:
                return await self.arun(q, **shared)

        return list(await asyncio.gather(*(_guarded(q) for q in queries)))

    # ------------------------------------------------------------------
    # Core sync execution
    # ------------------------------------------------------------------

    def _execute(self, user_query: str, cfg: dict[str, Any]) -> ClassifierResult:
        options: list[str] = cfg["options"]
        options_enum = _build_options_enum(options)

        self._check_rate_limit(cfg["user_id"])
        memory_ctx = self._get_memory_context(cfg["session_id"])

        usage = ClassifierUsage()
        prior_response = ""
        last_error = ""
        selected: list[str] = []
        confidences: list[float] | None = None
        all_scores: dict[str, float] | None = None
        reasoning: str | None = None
        fuzzy_corrected = False

        for attempt in range(self._max_retries + 1):
            message = _build_classifier_message(
                user_query=user_query,
                options=options,
                selection_mode=cfg["selection_mode"],
                include_confidence=self._include_confidence,
                include_reasoning=self._include_reasoning,
                score_all=self._score_all,
                option_descriptions=self._option_descriptions,
                memory_ctx=memory_ctx,
                prior_response=prior_response if attempt > 0 else "",
                error_msg=last_error if attempt > 0 else "",
            )

            start = time.perf_counter()
            chat_config = ChatConfig(
                user_message=message,
                prompt=self._prompt,
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
            )
            response = self._kit.chat(chat_config)
            latency_ms = (time.perf_counter() - start) * 1000

            resp_usage: dict[str, int] = response.usage or {}
            usage.input_tokens += resp_usage.get("prompt_tokens", 0)
            usage.output_tokens += resp_usage.get("completion_tokens", 0)
            if attempt > 0:
                usage.retry_count += 1

            content = response.content or ""
            self._record_span("classify", latency_ms, resp_usage)

            try:
                selected, confidences, all_scores, reasoning, fuzzy_corrected = (
                    _parse_and_validate(
                        content, options, options_enum,
                        cfg["selection_mode"],
                        self._include_confidence, self._include_reasoning,
                        self._score_all, self._case_sensitive,
                        cfg["confidence_threshold"], self._fuzzy_fallback,
                    )
                )
                break
            except ValueError as exc:
                last_error = str(exc)
                prior_response = content
                if attempt >= self._max_retries:
                    raise

        self._save_memory(cfg["session_id"], user_query, ", ".join(selected))
        self._emit_metrics(usage)

        return ClassifierResult(
            user_query=user_query,
            options_provided=options,
            selected=selected,
            confidences=confidences,
            all_scores=all_scores,
            reasoning=reasoning,
            fuzzy_corrected=fuzzy_corrected,
            uncertain=bool(self._uncertain_label and self._uncertain_label in selected),
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Core async execution
    # ------------------------------------------------------------------

    async def _aexecute(self, user_query: str, cfg: dict[str, Any]) -> ClassifierResult:
        options: list[str] = cfg["options"]
        options_enum = _build_options_enum(options)

        self._check_rate_limit(cfg["user_id"])
        memory_ctx = self._get_memory_context(cfg["session_id"])

        usage = ClassifierUsage()
        prior_response = ""
        last_error = ""
        selected: list[str] = []
        confidences: list[float] | None = None
        all_scores: dict[str, float] | None = None
        reasoning: str | None = None
        fuzzy_corrected = False

        for attempt in range(self._max_retries + 1):
            message = _build_classifier_message(
                user_query=user_query,
                options=options,
                selection_mode=cfg["selection_mode"],
                include_confidence=self._include_confidence,
                include_reasoning=self._include_reasoning,
                score_all=self._score_all,
                option_descriptions=self._option_descriptions,
                memory_ctx=memory_ctx,
                prior_response=prior_response if attempt > 0 else "",
                error_msg=last_error if attempt > 0 else "",
            )

            start = time.perf_counter()
            chat_config = ChatConfig(
                user_message=message,
                prompt=self._prompt,
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
            )
            response = await self._kit.achat(chat_config)
            latency_ms = (time.perf_counter() - start) * 1000

            resp_usage: dict[str, int] = response.usage or {}
            usage.input_tokens += resp_usage.get("prompt_tokens", 0)
            usage.output_tokens += resp_usage.get("completion_tokens", 0)
            if attempt > 0:
                usage.retry_count += 1

            content = response.content or ""
            self._record_span("classify", latency_ms, resp_usage)

            try:
                selected, confidences, all_scores, reasoning, fuzzy_corrected = (
                    _parse_and_validate(
                        content, options, options_enum,
                        cfg["selection_mode"],
                        self._include_confidence, self._include_reasoning,
                        self._score_all, self._case_sensitive,
                        cfg["confidence_threshold"], self._fuzzy_fallback,
                    )
                )
                break
            except ValueError as exc:
                last_error = str(exc)
                prior_response = content
                if attempt >= self._max_retries:
                    raise

        self._save_memory(cfg["session_id"], user_query, ", ".join(selected))
        self._emit_metrics(usage)

        return ClassifierResult(
            user_query=user_query,
            options_provided=options,
            selected=selected,
            confidences=confidences,
            all_scores=all_scores,
            reasoning=reasoning,
            fuzzy_corrected=fuzzy_corrected,
            uncertain=bool(self._uncertain_label and self._uncertain_label in selected),
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    def _resolve_config(  # noqa: PLR0913
        self,
        *,
        options: Any,
        selection_mode: str | None,
        output_format: str | None,
        temperature: float | None,
        max_tokens: int | None,
        confidence_threshold: Any,
        session_id: str | None,
        user_id: str | None,
    ) -> dict[str, Any]:
        """Merge pipeline-level defaults with per-call overrides."""
        with self._lock:
            base_options = list(self._options) if self._options is not None else None

        # Per-call options override
        resolved_options = base_options
        if options is not _UNSET:
            resolved_options = options

        if not resolved_options:
            raise ValueError(
                "options must be provided — either at construction time "
                "via ListClassifierPipeline(options=[...]) or per-call "
                "via run(options=[...])."
            )

        # Inject uncertain_label at the end (always last so it doesn't distort numbering)
        final_options = list(resolved_options)
        if self._uncertain_label and self._uncertain_label not in final_options:
            final_options.append(self._uncertain_label)

        resolved_threshold = self._confidence_threshold
        if confidence_threshold is not _UNSET:
            resolved_threshold = confidence_threshold

        return {
            "options":              final_options,
            "selection_mode":       selection_mode or self._selection_mode,
            "output_format":        output_format or self._output_format,
            "temperature":          temperature if temperature is not None else self._temperature,
            "max_tokens":           max_tokens if max_tokens is not None else self._max_tokens,
            "confidence_threshold": resolved_threshold,
            "session_id":           session_id,
            "user_id":              user_id if user_id is not None else self._user_id,
        }

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(
        result: ClassifierResult,
        output_format: str,
    ) -> ClassifierResult | str | dict[str, Any]:
        if output_format == "pydantic":
            return result
        if output_format == "string":
            return f"[error] {result.error}" if result.error else result.as_string()
        if output_format == "dict":
            return {"error": result.error, "selected": []} if result.error else result.as_dict()
        return result

    # ------------------------------------------------------------------
    # Pipeline-level caching
    # ------------------------------------------------------------------

    def _lookup_exact_cache(
        self, user_query: str, cfg: dict[str, Any]
    ) -> ClassifierResult | None:
        """Return a cached :class:`ClassifierResult` on an exact hit, else ``None``."""
        if self._exact_cache is None:
            return None
        key = _exact_cache_key(
            user_query, cfg["options"], cfg["selection_mode"],
            self._include_confidence, self._include_reasoning, self._score_all,
            cfg["temperature"], cfg["max_tokens"], cfg["confidence_threshold"],
        )
        try:
            cached = self._exact_cache.get(*key)
        except Exception:  # noqa: BLE001
            return None
        if cached is None:
            return None
        result = _llm_response_to_result(cached)
        if result is not None:
            result = result.model_copy(update={"cache_hit": "exact"})
        return result

    def _store_exact_cache(
        self, user_query: str, cfg: dict[str, Any], result: ClassifierResult
    ) -> None:
        """Store *result* in the exact cache (non-fatal on error)."""
        if self._exact_cache is None:
            return
        key = _exact_cache_key(
            user_query, cfg["options"], cfg["selection_mode"],
            self._include_confidence, self._include_reasoning, self._score_all,
            cfg["temperature"], cfg["max_tokens"], cfg["confidence_threshold"],
        )
        try:
            self._exact_cache.put(*key, _result_to_llm_response(result))
        except Exception:  # noqa: BLE001
            pass

    def _lookup_semantic_cache(self, user_query: str) -> ClassifierResult | None:
        """Return a cached :class:`ClassifierResult` on a semantic hit, else ``None``."""
        if self._semantic_cache is None:
            return None
        try:
            cached = self._semantic_cache.get(user_query)
        except Exception:  # noqa: BLE001
            return None
        if cached is None:
            return None
        result = _llm_response_to_result(cached)
        if result is not None:
            result = result.model_copy(update={"cache_hit": "semantic"})
        return result

    def _store_semantic_cache(self, user_query: str, result: ClassifierResult) -> None:
        """Store *result* in the semantic cache (non-fatal on error)."""
        if self._semantic_cache is None:
            return
        try:
            self._semantic_cache.put(user_query, _result_to_llm_response(result))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def _audit_log(
        self,
        result: ClassifierResult,
        *,
        session_id: str | None,
        user_id: str | None,
        latency_ms: float,
        timestamp: str,
    ) -> None:
        """Emit an :class:`AuditEntry` to the configured ``audit_logger`` (non-fatal)."""
        if self._audit_logger is None:
            return
        try:
            entry = result.to_audit_entry(
                timestamp=timestamp,
                user_id=user_id,
                session_id=session_id,
                latency_ms=latency_ms,
            )
            self._audit_logger.log(entry)
        except Exception:  # noqa: BLE001
            pass  # audit errors must never break the pipeline

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self, user_id: str | None) -> None:
        if self._rate_limiter is None or not user_id:
            return
        try:
            allowed: bool = self._rate_limiter.check_and_consume(user_id, 500)
        except Exception as exc:
            raise ClassifierRateLimitExceededError(
                f"Rate limiter error for user '{user_id}': {exc}"
            ) from exc
        if not allowed:
            try:
                remaining = self._rate_limiter.get_remaining(user_id)
            except Exception:  # noqa: BLE001
                remaining = 0
            raise ClassifierRateLimitExceededError(
                f"Rate limit exceeded for user '{user_id}'. "
                f"Remaining quota: {remaining} tokens."
            )

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _get_memory_context(self, session_id: str | None) -> str:
        if self._memory is None or not session_id:
            return ""
        try:
            history: list[dict[str, Any]] = self._memory.get_history(session_id)
        except Exception:  # noqa: BLE001
            return ""
        if not history:
            return ""
        parts = ["Prior conversation context:"]
        for turn in history[-6:]:
            role = str(turn.get("role", "unknown")).upper()
            content_snippet = str(turn.get("content", ""))[:300]
            parts.append(f"  {role}: {content_snippet}")
        return "\n".join(parts) + "\n\n"

    def _save_memory(
        self, session_id: str | None, user_query: str, answer: str
    ) -> None:
        if self._memory is None or not session_id:
            return
        try:
            self._memory.append(session_id, "user", user_query)
            if answer:
                self._memory.append(session_id, "assistant", answer)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _record_span(
        self, operation: str, latency_ms: float, usage: dict[str, int]
    ) -> None:
        if self._tracer is not None:
            self._tracer.record_chat_span(
                provider="pipeline",
                model=f"list_classifier.{operation}",
                latency_ms=latency_ms,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        if self._metrics is not None:
            self._metrics.record_request(
                provider="pipeline",
                model=f"list_classifier.{operation}",
                operation="chat",
                status="ok",
                latency_s=latency_ms / 1000,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

    def _emit_metrics(self, usage: ClassifierUsage) -> None:
        if self._metrics is not None:
            self._metrics.record_request(
                provider="pipeline",
                model="list_classifier",
                operation="pipeline",
                status="ok",
                latency_s=0.0,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )


# ---------------------------------------------------------------------------
# AsyncListClassifierPipeline — run() is async
# ---------------------------------------------------------------------------


class AsyncListClassifierPipeline(ListClassifierPipeline):
    """Async-first variant of :class:`ListClassifierPipeline`.

    ``run()`` is a coroutine — ``await pipeline.run(...)`` directly.
    Designed for FastAPI, aiohttp, Starlette, and other async frameworks.

    Constructor and all ``run()`` parameters are identical to
    :class:`ListClassifierPipeline`.

    Example
    -------
    ::

        pipeline = AsyncListClassifierPipeline.from_provider(
            "openai", "gpt-4o-mini",
            options=["Billing", "Support", "Sales"],
            safe_mode=True,
        )

        # FastAPI handler:
        @app.post("/classify")
        async def classify(query: str):
            result = await pipeline.run(query)
            return result.as_dict()
    """

    async def run(  # type: ignore[override]
        self,
        user_query: str,
        **kwargs: Any,
    ) -> ClassifierResult | str | dict[str, Any]:
        """Async ``run()`` — delegates to :meth:`ListClassifierPipeline.arun`."""
        return await self.arun(user_query, **kwargs)
