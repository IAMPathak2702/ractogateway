"""Cost-aware model router.

Dynamically selects the cheapest model that can handle the complexity of an
incoming request, without making an extra LLM call for classification.

Complexity scoring (pure heuristics, O(1) per call):

1. **Token estimate** — ``len(text) // 4`` gives a rough word/token count.
   Scaled to contribute 0-50 points.
2. **Keyword density** — checks the message (lowercased) for a curated set
   of *complexity keywords* (e.g. "analyze", "compare", "implement").
   Each unique keyword found adds points, up to 50.
3. Score is clamped to [0, 100].

The router then walks the ``tiers`` list (sorted ascending by ``max_score``)
and returns the ``model`` of the **first** tier whose ``max_score ≥ score``.
The last tier is always the fallback.

Thread-safety: the router has no mutable state after construction — all
methods are pure functions.  Safe to share across threads / coroutines.
"""

from __future__ import annotations

from ractogateway.routing._models import RoutingTier, _RoutingTierList

# ---------------------------------------------------------------------------
# Complexity keyword sets
# ---------------------------------------------------------------------------

# Keywords associated with *reasoning-heavy* tasks.
_COMPLEX_KEYWORDS: frozenset[str] = frozenset(
    {
        "analyze",
        "analyse",
        "compare",
        "contrast",
        "evaluate",
        "critique",
        "synthesize",
        "synthesise",
        "implement",
        "refactor",
        "architect",
        "design",
        "algorithm",
        "optimize",
        "optimise",
        "debug",
        "diagnose",
        "research",
        "explain in detail",
        "step by step",
        "step-by-step",
        "reason through",
        "think through",
        "trade-offs",
        "tradeoffs",
        "pros and cons",
    }
)

# Rough contribution per matched keyword (clamped later).
_POINTS_PER_KEYWORD: int = 8
# Maximum points from keywords alone.
_MAX_KEYWORD_POINTS: int = 50
# Maximum points from token estimate alone.
_MAX_TOKEN_POINTS: int = 50
# One "token" ≈ 4 characters; threshold where full token points kick in.
_TOKEN_SATURATION: int = 400  # ~100 words


class CostAwareRouter:
    """Routes LLM requests to the appropriate model tier based on message
    complexity — without making any extra API calls.

    Parameters
    ----------
    tiers:
        Ordered list of :class:`~ractogateway.routing.RoutingTier` objects,
        sorted **ascending** by ``max_score`` (cheapest first).
        The last tier's ``max_score`` should be ``100`` to act as fallback.

    Raises
    ------
    ValueError
        If ``tiers`` is empty or not sorted ascending by ``max_score``.

    Example — 3-tier OpenAI ladder::

        from ractogateway.routing import CostAwareRouter, RoutingTier

        router = CostAwareRouter([
            RoutingTier(model="gpt-4o-mini",  max_score=30),
            RoutingTier(model="gpt-4o",        max_score=70),
            RoutingTier(model="o3-mini",        max_score=100),
        ])

        model = router.route("What is 2+2?")         # → "gpt-4o-mini"
        model = router.route("Analyze the trade-offs between Redis Cluster and "
                             "Cassandra for a write-heavy time-series workload …")
        # → "o3-mini"

    Example — binary routing (2 tiers)::

        router = CostAwareRouter([
            RoutingTier(model="claude-haiku-4-5-20251001", max_score=40),
            RoutingTier(model="claude-opus-4-6",           max_score=100),
        ])
    """

    def __init__(self, tiers: list[RoutingTier]) -> None:
        # Validates non-empty + ascending sort via Pydantic model.
        validated = _RoutingTierList(tiers=tiers)
        self._tiers: tuple[RoutingTier, ...] = tuple(validated.tiers)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, text: str) -> int:
        """Compute a complexity score in [0, 100] for *text*.

        A higher score means a more complex task.

        Algorithm
        ---------
        token_pts = min(len(text)//4, SAT) * (MAX_TP / SAT)
        kw_pts    = min(matches * PPK, MAX_KP)
        score     = clamp(token_pts + kw_pts, 0, 100)
        """
        lower = text.lower()

        # Token-length contribution (0-50)
        token_est = len(text) // 4
        token_pts = min(token_est, _TOKEN_SATURATION) * _MAX_TOKEN_POINTS / _TOKEN_SATURATION

        # Keyword contribution (0-50)
        matches = sum(1 for kw in _COMPLEX_KEYWORDS if kw in lower)
        keyword_pts = min(matches * _POINTS_PER_KEYWORD, _MAX_KEYWORD_POINTS)

        raw = token_pts + keyword_pts
        return max(0, min(100, round(raw)))

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, text: str) -> str:
        """Return the model identifier for *text*.

        Walks tiers (cheapest first) and returns the first model whose
        ``max_score ≥ complexity_score``.  Always returns a model because the
        last tier has ``max_score == 100`` (validated at construction).

        Complexity: O(k) where k = number of tiers.
        """
        complexity = self.score(text)
        for tier in self._tiers:
            if complexity <= tier.max_score:
                return tier.model
        # Safety fallback — last tier always catches (guaranteed by validator).
        return self._tiers[-1].model  # pragma: no cover

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def tiers(self) -> tuple[RoutingTier, ...]:
        """Immutable view of the configured tiers."""
        return self._tiers

    def __repr__(self) -> str:  # pragma: no cover
        tier_str = ", ".join(f"{t.model}(≤{t.max_score})" for t in self._tiers)
        return f"CostAwareRouter([{tier_str}])"
