"""Chain-of-thought prompt injection utility.

Internal helper used by every developer kit when
``ChatConfig(chain_of_thought=True)`` is set.  The function is a pure
transformation — it never mutates the original prompt.
"""

from __future__ import annotations

from ractogateway.prompts.engine import RactoPrompt

_COT_CONSTRAINT = (
    "Before answering, reason through the problem step by step. "
    "State each reasoning step clearly and explicitly, then conclude with "
    "your final answer."
)


def apply_chain_of_thought(prompt: RactoPrompt) -> RactoPrompt:
    """Return a copy of *prompt* with a chain-of-thought constraint appended.

    The original prompt is never mutated.  The constraint is appended last so
    it appears at the bottom of the ``[CONSTRAINTS]`` section and is processed
    after all caller-defined rules.

    Parameters
    ----------
    prompt:
        The :class:`~ractogateway.prompts.engine.RactoPrompt` to augment.

    Returns
    -------
    RactoPrompt
        A new instance with the CoT constraint added.
    """
    return prompt.model_copy(
        update={"constraints": [*list(prompt.constraints), _COT_CONSTRAINT]}
    )
