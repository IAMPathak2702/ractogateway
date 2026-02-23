"""Shared Pydantic response-model validation helpers for all developer kits.

All three kits (OpenAI, Google, Anthropic) use identical validation and retry
logic.  Centralising it here means a bug fix or improvement applies to every
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
token-by-token to the caller).  :func:`validate_stream_final` raises
:class:`~ractogateway.exceptions.ResponseModelValidationError` immediately on
the final chunk if validation fails.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from ractogateway.adapters.base import LLMResponse, try_parse_json
from ractogateway.exceptions import ResponseModelValidationError

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_validation_errors(exc: ValidationError) -> str:
    """Convert a Pydantic ``ValidationError`` into a bulleted correction list.

    Each line names the field path, states what went wrong, and shows the
    offending value so the LLM knows exactly what to fix.
    """
    lines: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        msg = error["msg"]
        inp = error.get("input", "<unknown>")
        lines.append(f"  - {loc}: {msg} (your value: {inp!r})")
    return "\n".join(lines)


def _build_correction_message(bad_json: str, error_text: str, attempt: int) -> str:
    """Return the user message sent on each retry attempt.

    The message includes the bad JSON and the exact field errors so the LLM
    can produce a minimal, targeted correction.
    """
    return (
        f"Your previous JSON response (attempt {attempt}) failed schema validation.\n"
        "Fix ONLY the invalid fields listed below, keep all other fields unchanged,\n"
        "and return the complete corrected JSON with no explanation or markdown fences.\n\n"
        f"Validation errors:\n{error_text}\n\n"
        f"Your previous response:\n{bad_json}"
    )


# ---------------------------------------------------------------------------
# Public API — sync
# ---------------------------------------------------------------------------


def validate_and_retry(
    response: LLMResponse,
    config: Any,  # ChatConfig — avoid circular import
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
        A callable ``(correction_user_message: str) -> LLMResponse``.  The kit
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
    if config.response_model is None or not isinstance(response.parsed, dict):
        return response

    last_exc: ValidationError | None = None
    current = response

    for attempt in range(config.max_validation_retries + 1):
        try:
            validated = config.response_model.model_validate(current.parsed)
            current.parsed = validated.model_dump()
            return current
        except ValidationError as exc:
            last_exc = exc
            if attempt >= config.max_validation_retries:
                break

            # Build a correction prompt and re-call the adapter.
            error_text = _format_validation_errors(exc)
            bad_json = current.content or json.dumps(current.parsed, indent=2)
            correction = _build_correction_message(bad_json, error_text, attempt + 1)

            retry_response = adapter_run(correction)
            if isinstance(retry_response.parsed, dict):
                current = retry_response
            else:
                # Retry returned non-JSON — give up early.
                break

    raise ResponseModelValidationError(
        f"response_model validation failed after "
        f"{config.max_validation_retries + 1} attempt(s). "
        f"Last error: {last_exc}",
        attempts=config.max_validation_retries + 1,
        last_error=last_exc,  # type: ignore[arg-type]
        raw_response=current.content,
    )


# ---------------------------------------------------------------------------
# Public API — async
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
    if config.response_model is None or not isinstance(response.parsed, dict):
        return response

    last_exc: ValidationError | None = None
    current = response

    for attempt in range(config.max_validation_retries + 1):
        try:
            validated = config.response_model.model_validate(current.parsed)
            current.parsed = validated.model_dump()
            return current
        except ValidationError as exc:
            last_exc = exc
            if attempt >= config.max_validation_retries:
                break

            error_text = _format_validation_errors(exc)
            bad_json = current.content or json.dumps(current.parsed, indent=2)
            correction = _build_correction_message(bad_json, error_text, attempt + 1)

            retry_response = await adapter_arun(correction)
            if isinstance(retry_response.parsed, dict):
                current = retry_response
            else:
                break

    raise ResponseModelValidationError(
        f"response_model validation failed after "
        f"{config.max_validation_retries + 1} attempt(s). "
        f"Last error: {last_exc}",
        attempts=config.max_validation_retries + 1,
        last_error=last_exc,  # type: ignore[arg-type]
        raw_response=current.content,
    )


# ---------------------------------------------------------------------------
# Public API — streaming (no retry possible)
# ---------------------------------------------------------------------------


def validate_stream_final(
    accumulated_text: str,
    config: Any,  # ChatConfig
) -> Any:
    """Validate the final accumulated stream text against ``config.response_model``.

    Streaming cannot be retried because content is already delivered
    token-by-token.  On failure a
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
