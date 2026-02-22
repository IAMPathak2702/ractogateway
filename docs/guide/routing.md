# Cost-Aware Routing

`CostAwareRouter` dynamically routes each LLM request to the cheapest model capable of handling the task — without any extra API calls.

## How It Works

You define *tiers* sorted by ascending `max_score`. The router scores the incoming prompt (length, question complexity, keyword signals) and picks the first tier whose `max_score` is ≥ the computed score.

```python
from ractogateway.routing import CostAwareRouter, RoutingTier
from ractogateway import openai_developer_kit as gpt

router = CostAwareRouter([
    RoutingTier(model="gpt-4o-mini", max_score=30),   # simple / short prompts
    RoutingTier(model="gpt-4o",      max_score=70),   # moderate complexity
    RoutingTier(model="o3-mini",     max_score=100),  # complex / fallback
])

kit = gpt.OpenAIDeveloperKit(model="auto", router=router)

# Simple arithmetic → routed to gpt-4o-mini automatically
response = kit.chat(gpt.ChatConfig(user_message="What is 2 + 2?"))
```

Set `model="auto"` to enable routing. A `ValueError` is raised if `model="auto"` is used without a router.

## Cross-Provider Routing

Each provider kit accepts a `router` parameter, so the same pattern works for Google and Anthropic kits:

```python
from ractogateway import google_developer_kit as gemini
from ractogateway.routing import CostAwareRouter, RoutingTier

router = CostAwareRouter([
    RoutingTier(model="gemini-2.0-flash", max_score=40),
    RoutingTier(model="gemini-2.0-pro",   max_score=100),
])

kit = gemini.GoogleDeveloperKit(model="auto", router=router)
```

## Installation

Routing is included in the base install — no extra dependencies required.
