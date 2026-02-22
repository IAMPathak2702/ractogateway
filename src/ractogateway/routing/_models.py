"""Data models for the cost-aware routing subsystem."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class RoutingTier(BaseModel):
    """One tier in the cost-aware routing ladder.

    The router evaluates a complexity *score* (0-100) for each incoming
    message and selects the **first** tier whose ``max_score`` is >= that
    score.  The last tier in the list always acts as the catch-all fallback.

    Parameters
    ----------
    model:
        The LLM model identifier to use for requests that fall in this tier
        (e.g. ``"gpt-4o-mini"``, ``"gemini-2.0-flash"``, ``"claude-haiku-4-5-20251001"``).
    max_score:
        Inclusive upper bound on the complexity score that routes to this
        model.  Range: 0-100.  Set to ``100`` for the last (most powerful)
        tier so it catches everything.

    Examples
    --------
    ::

        tiers = [
            RoutingTier(model="gpt-4o-mini",  max_score=30),
            RoutingTier(model="gpt-4o",        max_score=70),
            RoutingTier(model="o3-mini",        max_score=100),
        ]
    """

    model: str = Field(min_length=1, description="Provider model identifier.")
    max_score: int = Field(
        default=100,
        ge=0,
        le=100,
        description="Inclusive upper-bound complexity score for this tier.",
    )


class _RoutingTierList(BaseModel):
    """Internal validator that enforces tiers are sorted ascending by max_score."""

    tiers: list[RoutingTier] = Field(min_length=1)

    @model_validator(mode="after")
    def _ensure_sorted(self) -> _RoutingTierList:
        scores = [t.max_score for t in self.tiers]
        if scores != sorted(scores):
            raise ValueError(
                "RoutingTier list must be sorted ascending by max_score "
                "(cheapest to most capable).  "
                f"Got scores: {scores}"
            )
        return self
