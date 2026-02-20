"""Cost-aware model routing for RactoGateway.

Dynamically routes LLM requests to the cheapest model capable of handling
the task complexity — without any extra API calls.

Quick start::

    from ractogateway.routing import CostAwareRouter, RoutingTier
    from ractogateway import openai_developer_kit as gpt

    router = CostAwareRouter([
        RoutingTier(model="gpt-4o-mini", max_score=30),   # simple tasks
        RoutingTier(model="gpt-4o",       max_score=70),   # moderate
        RoutingTier(model="o3-mini",       max_score=100),  # complex / fallback
    ])

    kit = gpt.OpenAIDeveloperKit(model="auto", router=router)
    response = kit.chat(gpt.ChatConfig(user_message="What is 2+2?"))
    # Routes to gpt-4o-mini automatically.
"""

from ractogateway.routing._models import RoutingTier
from ractogateway.routing.router import CostAwareRouter

__all__ = [
    "CostAwareRouter",
    "RoutingTier",
]
