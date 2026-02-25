"""Built-in model pricing table (USD per 1 million tokens).

Prices are approximate and may lag behind provider announcements.
Override or extend via ``price_table=`` on :class:`~ractogateway.telemetry.RactoTracer`
and :class:`~ractogateway.telemetry.GatewayMetricsMiddleware`.
"""

from __future__ import annotations

from ractogateway.telemetry._models import ModelPricing

#: Default pricing table.  Keys are model identifiers as returned by the provider.
DEFAULT_COST_TABLE: dict[str, ModelPricing] = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt-4o": ModelPricing(input_per_million=2.50, output_per_million=10.00),
    "gpt-4o-mini": ModelPricing(input_per_million=0.15, output_per_million=0.60),
    "gpt-4-turbo": ModelPricing(input_per_million=10.00, output_per_million=30.00),
    "gpt-4": ModelPricing(input_per_million=30.00, output_per_million=60.00),
    "gpt-3.5-turbo": ModelPricing(input_per_million=0.50, output_per_million=1.50),
    "o1": ModelPricing(input_per_million=15.00, output_per_million=60.00),
    "o1-mini": ModelPricing(input_per_million=3.00, output_per_million=12.00),
    "o3-mini": ModelPricing(input_per_million=1.10, output_per_million=4.40),
    # ── Anthropic ─────────────────────────────────────────────────────────────
    "claude-opus-4-6": ModelPricing(input_per_million=15.00, output_per_million=75.00),
    "claude-sonnet-4-6": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-haiku-4-5-20251001": ModelPricing(input_per_million=0.80, output_per_million=4.00),
    "claude-sonnet-4-5-20250929": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-3-5-sonnet-20241022": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-3-5-haiku-20241022": ModelPricing(input_per_million=0.80, output_per_million=4.00),
    "claude-3-opus-20240229": ModelPricing(input_per_million=15.00, output_per_million=75.00),
    "claude-3-haiku-20240307": ModelPricing(input_per_million=0.25, output_per_million=1.25),
    # ── Google Gemini ─────────────────────────────────────────────────────────
    "gemini-2.0-flash": ModelPricing(input_per_million=0.10, output_per_million=0.40),
    "gemini-2.0-flash-lite": ModelPricing(input_per_million=0.075, output_per_million=0.30),
    "gemini-2.5-pro": ModelPricing(input_per_million=1.25, output_per_million=10.00),
    "gemini-1.5-pro": ModelPricing(input_per_million=1.25, output_per_million=5.00),
    "gemini-1.5-flash": ModelPricing(input_per_million=0.075, output_per_million=0.30),
    "gemini-1.5-flash-8b": ModelPricing(input_per_million=0.0375, output_per_million=0.15),
    "gemini-1.0-pro": ModelPricing(input_per_million=0.50, output_per_million=1.50),
}


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    extra_table: dict[str, ModelPricing] | None = None,
) -> float:
    """Compute the estimated USD cost for a single LLM call.

    Parameters
    ----------
    model:
        Model identifier (e.g. ``"gpt-4o"``).  If not found in the
        combined table the function returns ``0.0``.
    input_tokens:
        Number of prompt tokens consumed.
    output_tokens:
        Number of completion tokens produced.
    extra_table:
        Optional ``{model: ModelPricing}`` dict to override or extend
        :data:`DEFAULT_COST_TABLE`.  Extra entries win over defaults.

    Returns
    -------
    float
        Estimated cost in USD, or ``0.0`` when the model is unknown.
    """
    table: dict[str, ModelPricing] = {**DEFAULT_COST_TABLE, **(extra_table or {})}
    pricing = table.get(model)
    if pricing is None:
        return 0.0
    return (
        input_tokens * pricing.input_per_million / 1_000_000
        + output_tokens * pricing.output_per_million / 1_000_000
    )
