"""ractogateway.prompts — RACTO Prompt Engine.

Provides the ``RactoPrompt`` Pydantic model that enforces structured prompt
definition via the RACTO principle (Role, Aim, Constraints, Tone, Output)
and compiles it into an optimized, anti-hallucination system prompt.
"""

from ractogateway.prompts.engine import RactoPrompt

__all__ = ["RactoPrompt"]
