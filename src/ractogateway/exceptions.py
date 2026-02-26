"""Unified exception hierarchy for RactoGateway.

All provider-specific SDK errors (``openai.APITimeoutError``,
``anthropic.APITimeoutError``, ``google.api_core.exceptions.*``, etc.) are
caught by the adapters and re-raised as one of these classes so callers
never need to import provider packages just to handle errors.

Example::

    from ractogateway.exceptions import RactoGatewayTimeoutError

    try:
        response = kit.chat(config)
    except RactoGatewayTimeoutError:
        # Retry or surface a user-friendly message.
        ...
"""

from __future__ import annotations


class RactoGatewayError(Exception):
    """Base class for every RactoGateway runtime error."""


class RactoGatewayTimeoutError(RactoGatewayError):
    """The upstream provider did not respond within the allowed time."""


class RactoGatewayAPIError(RactoGatewayError):
    """The upstream provider returned an error response.

    Attributes
    ----------
    status_code:
        HTTP status code returned by the provider, when available.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RactoGatewayAuthError(RactoGatewayError):
    """API key missing, invalid, or not authorised for the requested resource."""


class ResponseModelValidationError(RactoGatewayError):
    """Raised when ``response_model`` validation cannot be satisfied.

    The LLM returned structurally valid JSON but Pydantic rejected it (e.g.
    a field value was out of range) and all automatic retry attempts were
    exhausted.

    Attributes
    ----------
    attempts:
        Total number of API calls made (1 initial + N retries).
    last_error:
        The final :class:`pydantic.ValidationError` that triggered this
        exception.
    raw_response:
        The LLM's raw text from the last attempt, available for inspection
        or manual recovery.

    Example::

        from ractogateway.exceptions import ResponseModelValidationError

        try:
            response = kit.chat(config)
        except ResponseModelValidationError as e:
            print(e.last_error)       # Pydantic ValidationError details
            print(e.raw_response)     # raw JSON string from the LLM
            print(e.attempts)         # how many times we tried
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: Exception,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.raw_response = raw_response


# ---------------------------------------------------------------------------
# Internal helper — not part of the public API surface but importable.
# ---------------------------------------------------------------------------

_TIMEOUT_MARKERS: frozenset[str] = frozenset({"Timeout", "timeout", "DeadlineExceeded"})
_AUTH_MARKERS: frozenset[str] = frozenset(
    {"Authentication", "Unauthorized", "Forbidden", "PermissionDenied", "Unauthenticated"}
)


def _wrap_provider_error(exc: Exception, provider: str) -> Exception:
    """Classify a raw provider SDK exception as a :class:`RactoGatewayError`.

    Uses type-name inspection so that provider packages (``openai``,
    ``anthropic``, ``google-genai``) do not need to be imported at the
    module level of any adapter.

    Parameters
    ----------
    exc:
        The original exception from the provider SDK.
    provider:
        Short provider name used in the error message (e.g. ``"openai"``).

    Returns
    -------
    Exception
        A normalised :class:`RactoGatewayError` for provider-originated
        exceptions, or the original exception when it is a plain built-in
        Python error.
    """
    # Preserve built-in exceptions (e.g. RuntimeError injected by tests or
    # programmer errors) so they are not masked as provider API failures.
    if type(exc).__module__ == "builtins":
        return exc

    type_name = type(exc).__name__
    message = f"[{provider}] {exc}"

    if any(marker in type_name for marker in _TIMEOUT_MARKERS):
        return RactoGatewayTimeoutError(message)

    if any(marker in type_name for marker in _AUTH_MARKERS):
        return RactoGatewayAuthError(message)

    raw_status = getattr(exc, "status_code", None)
    status_code: int | None = raw_status if isinstance(raw_status, int) else None
    return RactoGatewayAPIError(message, status_code=status_code)
