"""RACTO Prompt Engine — structured, anti-hallucination prompt compilation.

The ``RactoPrompt`` model enforces the RACTO principle:

    **R** ole  — Who the model is.
    **A** im   — What it must accomplish.
    **C** onstraints — Hard boundaries it must never violate.
    **T** one  — Communication style.
    **O** utput — The exact shape of the expected response.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sentinel for "no default"
# ---------------------------------------------------------------------------

class _Unset:
    """Internal sentinel — distinguishes 'user passed None' from 'not set'."""


# ---------------------------------------------------------------------------
# Output format helpers
# ---------------------------------------------------------------------------

def _schema_from_model(model: type[BaseModel]) -> dict[str, Any]:
    """Extract a clean JSON Schema dict from a Pydantic v2 model."""
    schema = model.model_json_schema()
    # Remove pydantic-internal metadata that clutters the prompt.
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


def _render_output_block(output_format: str | type[BaseModel]) -> str:
    """Return the OUTPUT section content for the compiled prompt."""
    if isinstance(output_format, type) and issubclass(output_format, BaseModel):
        schema = _schema_from_model(output_format)
        schema_json = json.dumps(schema, indent=2)
        return (
            "Respond ONLY with valid JSON that conforms exactly to the "
            "following JSON Schema. Do NOT wrap the JSON in markdown code "
            "fences or add any text before or after it.\n\n"
            f"JSON Schema:\n{schema_json}"
        )

    tag = output_format.strip().lower()

    if tag == "json":
        return (
            "Respond ONLY with valid JSON. Do NOT wrap the response in "
            "markdown code fences (```json … ```) or add any commentary "
            "before or after the JSON object."
        )

    if tag == "markdown":
        return "Respond in well-structured Markdown."

    if tag == "text":
        return "Respond in plain text with no special formatting."

    # Free-form format description provided by the user.
    return f"Respond using the following format:\n{output_format}"


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class RactoPrompt(BaseModel):
    """A strictly validated RACTO prompt definition.

    Parameters
    ----------
    role:
        A sentence (or short paragraph) describing **who** the LLM is.
    aim:
        A clear statement of the task objective.
    constraints:
        Hard rules the model must obey.  At least one is required.
    tone:
        The desired communication style (e.g. "Professional and concise").
    output_format:
        Either a format keyword (``"json"``, ``"text"``, ``"markdown"``),
        a free-form format description, or a **Pydantic model class** whose
        JSON Schema will be embedded in the prompt.
    context:
        Optional extra context paragraph injected between AIM and
        CONSTRAINTS.  Useful for passing domain-specific background
        knowledge that the model needs to reason about.
    examples:
        Optional list of example input/output pairs that are included in
        the prompt to steer the model via few-shot learning.
    anti_hallucination:
        When *True* (the default), the compiler appends explicit
        anti-hallucination directives at the end of the prompt.
    """

    role: str = Field(
        ...,
        min_length=1,
        description="Who the model is (e.g. 'You are a senior Python engineer').",
    )
    aim: str = Field(
        ...,
        min_length=1,
        description="A clear statement of the task objective.",
    )
    constraints: list[str] = Field(
        ...,
        min_length=1,
        description="Hard rules the model must obey. Minimum one constraint.",
    )
    tone: str = Field(
        ...,
        min_length=1,
        description="Desired communication style.",
    )
    output_format: Union[str, type[BaseModel]] = Field(  # noqa: UP007
        ...,
        description=(
            "A format keyword ('json', 'text', 'markdown'), a free-form "
            "description, or a Pydantic BaseModel class."
        ),
    )
    context: str | None = Field(
        default=None,
        description="Optional domain-specific background knowledge.",
    )
    examples: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Optional few-shot examples. Each dict should have "
            "'input' and 'output' keys."
        ),
    )
    anti_hallucination: bool = Field(
        default=True,
        description="Append anti-hallucination directives to the prompt.",
    )

    # Allow arbitrary types so that `type[BaseModel]` passes validation.
    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_constraints_not_empty_strings(self) -> "RactoPrompt":
        for idx, c in enumerate(self.constraints):
            if not c.strip():
                raise ValueError(
                    f"constraints[{idx}] is blank. Every constraint must be "
                    "a non-empty string."
                )
        return self

    @model_validator(mode="after")
    def _validate_examples_shape(self) -> "RactoPrompt":
        if self.examples is not None:
            for idx, ex in enumerate(self.examples):
                if "input" not in ex or "output" not in ex:
                    raise ValueError(
                        f"examples[{idx}] must contain both 'input' and "
                        f"'output' keys. Got: {sorted(ex.keys())}"
                    )
        return self

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self) -> str:
        """Compile the RACTO fields into an optimized system prompt string.

        The resulting prompt is structured into clearly delimited sections
        so that the LLM can parse each instruction block unambiguously.

        Returns
        -------
        str
            A ready-to-use system prompt.
        """
        sections: list[str] = []

        # --- ROLE ---
        sections.append(f"[ROLE]\n{self.role}")

        # --- AIM ---
        sections.append(f"[AIM]\n{self.aim}")

        # --- CONTEXT (optional) ---
        if self.context:
            sections.append(f"[CONTEXT]\n{self.context}")

        # --- CONSTRAINTS ---
        constraint_lines = "\n".join(
            f"- {c}" for c in self.constraints
        )
        sections.append(f"[CONSTRAINTS]\n{constraint_lines}")

        # --- TONE ---
        sections.append(f"[TONE]\n{self.tone}")

        # --- OUTPUT ---
        output_block = _render_output_block(self.output_format)
        sections.append(f"[OUTPUT]\n{output_block}")

        # --- EXAMPLES (optional) ---
        if self.examples:
            example_parts: list[str] = []
            for i, ex in enumerate(self.examples, start=1):
                example_parts.append(
                    f"Example {i}:\n"
                    f"  Input:  {ex['input']}\n"
                    f"  Output: {ex['output']}"
                )
            sections.append("[EXAMPLES]\n" + "\n\n".join(example_parts))

        # --- ANTI-HALLUCINATION FOOTER ---
        if self.anti_hallucination:
            sections.append(
                textwrap.dedent("""\
                    [GUARDRAILS]
                    - If you are unsure or lack sufficient information, state it explicitly rather than guessing.
                    - Do NOT fabricate facts, citations, URLs, statistics, or code that you cannot verify.
                    - Stick strictly to what is asked. Do not add unrequested information.
                    - If the answer requires assumptions, list each assumption explicitly before proceeding.""")
            )

        return "\n\n".join(sections) + "\n"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_messages(
        self,
        user_message: str,
        *,
        provider: str = "generic",
    ) -> list[dict[str, str]]:
        """Return a ready-to-send message list for a given LLM provider.

        Parameters
        ----------
        user_message:
            The end-user's query or input.
        provider:
            One of ``"openai"``, ``"anthropic"``, ``"google"``, or
            ``"generic"``.  Controls the system-role key name.

        Returns
        -------
        list[dict[str, str]]
            A list of message dicts suitable for the provider's API.
        """
        system_prompt = self.compile()

        if provider in ("openai", "generic"):
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        if provider == "anthropic":
            # Anthropic uses "system" as a top-level param, but for message
            # list representation we use the same structure — the adapter
            # will unpack it.
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        if provider == "google":
            # Gemini separates system_instruction from contents.
            # The adapter will split this; we use a marker role.
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Expected one of: 'openai', 'anthropic', 'google', 'generic'."
        )

    def __str__(self) -> str:
        """Return the compiled prompt when cast to str."""
        return self.compile()
