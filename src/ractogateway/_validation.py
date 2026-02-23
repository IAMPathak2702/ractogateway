"""Shared Pydantic response-model validation helpers for all developer kits.

All three kits (OpenAI, Google, Anthropic) use identical validation and retry
logic. Centralizing it here means a bug fix or improvement applies to every
provider at once.

Validation strategy
-------------------
1. Attempt ``response_model.model_validate(response.parsed)``.
2. On :class:`pydantic.ValidationError`, format the field-level errors into a
   plain-English correction prompt that includes the bad JSON.
3. Call ``adapter_run(correction_msg)`` to get a fresh LLM response.
4. Repeat up to ``config.max_validation_retries`` times.
5. If still failing, raise :class:`~ractogateway.exceptions.ResponseModelValidationError`
   with the last error and raw response attached.

Streaming note
--------------
Streaming responses cannot be retried (the content has already been delivered
token-by-token to the caller). :func:`validate_stream_final` raises
:class:`~ractogateway.exceptions.ResponseModelValidationError` immediately on
the final chunk if validation fails.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError

from ractogateway.adapters.base import LLMResponse, try_parse_json
from ractogateway.exceptions import ResponseModelValidationError

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_TITLE_FIELD_NAMES: frozenset[str] = frozenset(
    {"title", "movie_title", "book_title", "subject", "name"}
)
_QUOTED_TEXT_RE = re.compile(r"['\"]([^'\"]+)['\"]")
_MEDIA_SUBJECT_RE = re.compile(
    r"\b(?:movie|film|book|song|series|show)\s+([A-Za-z0-9][^.!?,;:\n]{0,120})",
    re.IGNORECASE,
)


def _loc_to_path(loc: tuple[Any, ...]) -> str:
    """Return a printable dotted location path for a validation error."""
    path = ".".join(str(part) for part in loc)
    return path or "<root>"


def _is_missing_error(error: Mapping[str, Any]) -> bool:
    """Return True when a Pydantic error item denotes a missing field."""
    err_type = str(error.get("type", ""))
    return err_type == "missing" or err_type.endswith(".missing")


def _missing_error_paths(exc: ValidationError) -> list[str]:
    """Collect missing-field paths from a ValidationError."""
    paths: list[str] = []
    for error in exc.errors():
        if _is_missing_error(error):
            raw_loc = error.get("loc", ())
            loc = tuple(raw_loc) if isinstance(raw_loc, (tuple, list)) else ()
            paths.append(_loc_to_path(loc))
    return paths


def _required_field_names(response_model: Any) -> list[str]:
    """Best-effort extraction of required field names from a Pydantic model type."""
    fields = getattr(response_model, "model_fields", {})
    if not isinstance(fields, dict):
        return []
    return list(fields.keys())


def _ensure_dict_payload(response: LLMResponse) -> dict[str, Any] | None:
    """Ensure ``response.parsed`` is a dict, reparsing ``response.content`` if needed."""
    if isinstance(response.parsed, dict):
        return response.parsed
    if isinstance(response.content, str):
        reparsed = try_parse_json(response.content)
        if isinstance(reparsed, dict):
            response.parsed = reparsed
            return reparsed
    return None


def with_inferred_response_model(config: Any, prompt: Any) -> Any:
    """Return config with inferred ``response_model`` when prompt uses a model output.

    If ``config.response_model`` is unset and ``prompt.output_format`` is a
    Pydantic model class, we infer that same model for validation/retries.
    """
    if getattr(config, "response_model", None) is not None:
        return config

    output_format = getattr(prompt, "output_format", None)
    if not (isinstance(output_format, type) and issubclass(output_format, BaseModel)):
        return config

    copier = getattr(config, "model_copy", None)
    if callable(copier):
        return copier(update={"response_model": output_format})

    return config


def _extract_subject_from_user_message(user_message: str) -> str | None:
    """Best-effort extraction of a subject/title phrase from user input."""
    quoted = _QUOTED_TEXT_RE.search(user_message)
    if quoted:
        value = quoted.group(1).strip()
        if value:
            return value

    media_match = _MEDIA_SUBJECT_RE.search(user_message)
    if media_match:
        value = media_match.group(1).strip(" .,!?:;\"'")
        if value:
            return value
    return None


def _field_min_length(field_info: Any) -> int:
    """Return min_length constraint from Pydantic field metadata when present."""
    min_len = 0
    for meta in getattr(field_info, "metadata", ()):
        candidate = getattr(meta, "min_length", None)
        if isinstance(candidate, int) and candidate > min_len:
            min_len = candidate
    return min_len


def _infer_number_default(field_info: Any, *, as_int: bool) -> int | float:
    """Infer a constraint-compliant numeric fallback from field metadata."""
    lower_bound: int | float | None = None
    upper_bound: int | float | None = None

    for meta in getattr(field_info, "metadata", ()):
        ge = getattr(meta, "ge", None)
        gt = getattr(meta, "gt", None)
        le = getattr(meta, "le", None)
        lt = getattr(meta, "lt", None)

        if isinstance(ge, (int, float)):
            lower_bound = max(lower_bound, ge) if lower_bound is not None else ge
        if isinstance(gt, (int, float)):
            # Shift by the smallest representable practical step.
            bump = 1 if as_int else 1e-6
            candidate = gt + bump
            lower_bound = max(lower_bound, candidate) if lower_bound is not None else candidate
        if isinstance(le, (int, float)):
            upper_bound = min(upper_bound, le) if upper_bound is not None else le
        if isinstance(lt, (int, float)):
            bump = 1 if as_int else 1e-6
            candidate = lt - bump
            upper_bound = min(upper_bound, candidate) if upper_bound is not None else candidate

    value = lower_bound if lower_bound is not None else 0

    if upper_bound is not None and value > upper_bound:
        value = upper_bound

    if as_int:
        return int(value)
    return float(value)


def _unwrap_optional_annotation(annotation: Any) -> Any:
    """Unwrap Optional[T] / T | None into T when possible."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is not None and args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _string_placeholder(field_name: str, field_info: Any, user_message: str) -> str:
    """Build a placeholder string honoring min_length and subject hints."""
    min_len = _field_min_length(field_info)
    inferred = (
        _extract_subject_from_user_message(user_message)
        if field_name.lower() in _TITLE_FIELD_NAMES
        else None
    )
    value = inferred or "unknown"
    if len(value) < min_len:
        value = value.ljust(min_len, "x") if inferred else "x" * min_len
    return value


def _list_placeholder(field_info: Any, item_annotation: Any) -> list[Any]:
    """Build a list placeholder honoring min_length and item annotation."""
    min_len = _field_min_length(field_info)
    if min_len <= 0:
        return []
    if item_annotation is str:
        return ["unknown"] * min_len
    if item_annotation is int:
        return [0] * min_len
    if item_annotation is float:
        return [0.0] * min_len
    if item_annotation is bool:
        return [False] * min_len
    return [{} for _ in range(min_len)]


def _placeholder_for_field(
    field_name: str,
    field_info: Any,
    user_message: str,
) -> Any:
    """Build a best-effort placeholder for one missing required field."""
    annotation = _unwrap_optional_annotation(getattr(field_info, "annotation", Any))
    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is str:
        return _string_placeholder(field_name, field_info, user_message)
    if annotation in (bool, int, float):
        numeric_or_bool: dict[Any, Any] = {
            bool: False,
            int: _infer_number_default(field_info, as_int=True),
            float: _infer_number_default(field_info, as_int=False),
        }
        return numeric_or_bool[annotation]
    if annotation is list or origin is list:
        item_ann = _unwrap_optional_annotation(args[0]) if args else str
        return _list_placeholder(field_info, item_ann)
    if annotation is dict or origin is dict:
        return {}

    # Last resort for unknown annotations.
    return "unknown"


def _top_level_missing_key(error: Mapping[str, Any]) -> str | None:
    """Return top-level key for a missing-field error, else None."""
    raw_loc = error.get("loc", ())
    if not isinstance(raw_loc, (tuple, list)) or len(raw_loc) != 1:
        return None
    key = raw_loc[0]
    return key if isinstance(key, str) else None


def _try_autofill_missing_required_fields(
    payload: dict[str, Any],
    *,
    response_model: Any,
    user_message: str,
    exc: ValidationError,
) -> dict[str, Any] | None:
    """Attempt deterministic autofill for missing top-level required fields.

    This fallback is intentionally conservative:
    - only missing-field errors are considered;
    - only top-level fields are autofilled;
    - final output must still pass full Pydantic validation.
    """
    errors = exc.errors()
    all_missing = bool(errors) and all(_is_missing_error(error) for error in errors)
    fields = getattr(response_model, "model_fields", {})
    if not all_missing or not isinstance(fields, dict):
        return None

    patched: dict[str, Any] = deepcopy(payload)
    missing_keys: list[str] = []
    for error in errors:
        key = _top_level_missing_key(error)
        if key is None or key not in fields:
            return None
        missing_keys.append(key)

    for key in missing_keys:
        if key not in patched:
            patched[key] = _placeholder_for_field(key, fields[key], user_message)

    try:
        validated = response_model.model_validate(patched)
    except ValidationError:
        return None
    dumped = validated.model_dump()
    if isinstance(dumped, dict):
        return dumped
    return None


def _format_validation_errors(exc: ValidationError) -> str:
    """Convert a Pydantic ``ValidationError`` into a bulleted correction list.

    Missing-field errors show an explicit "add this required field" directive
    instead of echoing the parent dict as "your value".
    """
    lines: list[str] = []
    for error in exc.errors():
        raw_loc = error.get("loc", ())
        loc_tuple = tuple(raw_loc) if isinstance(raw_loc, (tuple, list)) else ()
        loc = _loc_to_path(loc_tuple)
        msg = error["msg"]
        if _is_missing_error(error):
            lines.append(
                f"  - {loc}: {msg}. This required field is missing from your response; "
                "add it with a valid value."
            )
        else:
            inp = error.get("input", "<unknown>")
            lines.append(f"  - {loc}: {msg} (your value: {inp!r})")
    return "\n".join(lines)


def _build_correction_message(
    bad_json: str,
    error_text: str,
    attempt: int,
    *,
    model_name: str | None = None,
    required_fields: list[str] | None = None,
    missing_fields: list[str] | None = None,
) -> str:
    """Return the user message sent on each retry attempt.

    The message includes the bad JSON and the exact field errors so the LLM
    can produce a minimal, targeted correction.
    """
    requirements_hint = ""
    if model_name and required_fields:
        requirements_hint = (
            f"Required fields for {model_name}: {', '.join(required_fields)}.\n"
        )
    missing_hint = ""
    if missing_fields:
        missing_hint = (
            f"Missing required fields to add now: {', '.join(missing_fields)}.\n"
        )
    return (
        f"Your previous JSON response (attempt {attempt}) failed schema validation.\n"
        "Correct ONLY the fields listed below. If a required field is missing, add it.\n"
        f"{requirements_hint}"
        f"{missing_hint}"
        "Keep all other fields unchanged, and return the complete corrected JSON with no\n"
        "explanation or markdown fences.\n\n"
        f"Validation errors:\n{error_text}\n\n"
        f"Your previous response:\n{bad_json}"
    )


# ---------------------------------------------------------------------------
# Public API - sync
# ---------------------------------------------------------------------------


def validate_and_retry(
    response: LLMResponse,
    config: Any,  # ChatConfig - avoid circular import
    *,
    adapter_run: Callable[[str], LLMResponse],
) -> LLMResponse:
    """Validate *response* against ``config.response_model``, retrying on failure.

    Parameters
    ----------
    response:
        The initial :class:`~ractogateway.adapters.base.LLMResponse` from the
        provider API.
    config:
        A ``ChatConfig`` with ``response_model`` and ``max_validation_retries``
        fields.
    adapter_run:
        A callable ``(correction_user_message: str) -> LLMResponse``. The kit
        creates this closure to carry the original prompt, model, temperature,
        and extra kwargs so retries use the same provider settings.

    Returns
    -------
    LLMResponse
        The response with ``.parsed`` replaced by the validated Pydantic model
        dump on success.

    Raises
    ------
    ResponseModelValidationError
        When all retry attempts are exhausted and Pydantic still rejects the
        output.
    """
    if config.response_model is None:
        return response

    # If parsed payload is missing but content is JSON, recover it first.
    if _ensure_dict_payload(response) is None:
        return response

    last_exc: Exception | None = None
    current = response
    model_name = getattr(config.response_model, "__name__", "response model")
    required_fields = _required_field_names(config.response_model)

    for attempt in range(config.max_validation_retries + 1):
        try:
            payload = _ensure_dict_payload(current)
            if payload is None:
                break
            validated = config.response_model.model_validate(payload)
            current.parsed = validated.model_dump()
            return current
        except ValidationError as exc:
            last_exc = exc
            if attempt >= config.max_validation_retries:
                break

            # Build a correction prompt and re-call the adapter.
            error_text = _format_validation_errors(exc)
            missing_fields = _missing_error_paths(exc)
            bad_json = current.content or json.dumps(current.parsed, indent=2)
            correction = _build_correction_message(
                bad_json,
                error_text,
                attempt + 1,
                model_name=model_name,
                required_fields=required_fields,
                missing_fields=missing_fields,
            )

            retry_response = adapter_run(correction)
            if _ensure_dict_payload(retry_response) is not None:
                current = retry_response
            else:
                # Retry returned non-JSON - give up early.
                break

    if last_exc is None:
        last_exc = ValueError(
            "response_model validation failed because the response was not a JSON object."
        )

    payload = _ensure_dict_payload(current)
    if payload is not None and isinstance(last_exc, ValidationError):
        recovered = _try_autofill_missing_required_fields(
            payload,
            response_model=config.response_model,
            user_message=getattr(config, "user_message", ""),
            exc=last_exc,
        )
        if recovered is not None:
            current.parsed = recovered
            if current.content is None:
                current.content = json.dumps(recovered)
            return current

    raise ResponseModelValidationError(
        f"response_model validation failed after "
        f"{config.max_validation_retries + 1} attempt(s). "
        f"Last error: {last_exc}",
        attempts=config.max_validation_retries + 1,
        last_error=last_exc,
        raw_response=current.content,
    )


# ---------------------------------------------------------------------------
# Public API - async
# ---------------------------------------------------------------------------


async def async_validate_and_retry(
    response: LLMResponse,
    config: Any,  # ChatConfig
    *,
    adapter_arun: Callable[[str], Awaitable[LLMResponse]],
) -> LLMResponse:
    """Async variant of :func:`validate_and_retry`.

    Parameters
    ----------
    adapter_arun:
        An *async* callable ``async (correction_user_message: str) -> LLMResponse``.
    """
    if config.response_model is None:
        return response

    if _ensure_dict_payload(response) is None:
        return response

    last_exc: Exception | None = None
    current = response
    model_name = getattr(config.response_model, "__name__", "response model")
    required_fields = _required_field_names(config.response_model)

    for attempt in range(config.max_validation_retries + 1):
        try:
            payload = _ensure_dict_payload(current)
            if payload is None:
                break
            validated = config.response_model.model_validate(payload)
            current.parsed = validated.model_dump()
            return current
        except ValidationError as exc:
            last_exc = exc
            if attempt >= config.max_validation_retries:
                break

            error_text = _format_validation_errors(exc)
            missing_fields = _missing_error_paths(exc)
            bad_json = current.content or json.dumps(current.parsed, indent=2)
            correction = _build_correction_message(
                bad_json,
                error_text,
                attempt + 1,
                model_name=model_name,
                required_fields=required_fields,
                missing_fields=missing_fields,
            )

            retry_response = await adapter_arun(correction)
            if _ensure_dict_payload(retry_response) is not None:
                current = retry_response
            else:
                break

    if last_exc is None:
        last_exc = ValueError(
            "response_model validation failed because the response was not a JSON object."
        )

    payload = _ensure_dict_payload(current)
    if payload is not None and isinstance(last_exc, ValidationError):
        recovered = _try_autofill_missing_required_fields(
            payload,
            response_model=config.response_model,
            user_message=getattr(config, "user_message", ""),
            exc=last_exc,
        )
        if recovered is not None:
            current.parsed = recovered
            if current.content is None:
                current.content = json.dumps(recovered)
            return current

    raise ResponseModelValidationError(
        f"response_model validation failed after "
        f"{config.max_validation_retries + 1} attempt(s). "
        f"Last error: {last_exc}",
        attempts=config.max_validation_retries + 1,
        last_error=last_exc,
        raw_response=current.content,
    )


# ---------------------------------------------------------------------------
# Public API - streaming (no retry possible)
# ---------------------------------------------------------------------------


def validate_stream_final(
    accumulated_text: str,
    config: Any,  # ChatConfig
) -> Any:
    """Validate the final accumulated stream text against ``config.response_model``.

    Streaming cannot be retried because content is already delivered
    token-by-token. On failure a
    :class:`~ractogateway.exceptions.ResponseModelValidationError` is raised
    so callers get a clear, actionable error instead of silently receiving
    invalid data.

    Parameters
    ----------
    accumulated_text:
        The full streamed text concatenated across all chunks.
    config:
        ``ChatConfig`` with ``response_model``.

    Returns
    -------
    Any
        The validated Pydantic model dump (dict) on success, or the raw parsed
        value when ``response_model`` is ``None``.

    Raises
    ------
    ResponseModelValidationError
        When ``response_model`` is set and validation fails.
    """
    parsed = try_parse_json(accumulated_text)
    if config.response_model is None:
        return parsed
    if not isinstance(parsed, dict):
        return parsed
    try:
        validated = config.response_model.model_validate(parsed)
        return validated.model_dump()
    except ValidationError as exc:
        raise ResponseModelValidationError(
            f"response_model validation failed on stream final chunk. "
            f"Error: {exc}",
            attempts=1,
            last_error=exc,
            raw_response=accumulated_text,
        ) from exc

