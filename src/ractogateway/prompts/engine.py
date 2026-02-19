"""RACTO Prompt Engine — structured, anti-hallucination prompt compilation.

The ``RactoPrompt`` model enforces the RACTO principle:

    **R** ole  — Who the model is.
    **A** im   — What it must accomplish.
    **C** onstraints — Hard boundaries it must never violate.
    **T** one  — Communication style.
    **O** utput — The exact shape of the expected response.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import textwrap
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Sentinel for "no default"
# ---------------------------------------------------------------------------


class _Unset:
    """Internal sentinel — distinguishes 'user passed None' from 'not set'."""


# ---------------------------------------------------------------------------
# File attachment support
# ---------------------------------------------------------------------------

#: MIME types treated as images by every provider.
_IMAGE_MIMES: frozenset[str] = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


class RactoFile:
    """A file attachment that can be passed to :meth:`RactoPrompt.to_messages`.

    Create from a file path (MIME type is auto-detected) or directly from
    raw bytes with an explicit MIME type.

    Parameters
    ----------
    data:
        Raw bytes of the file.
    mime_type:
        MIME type string, e.g. ``"image/jpeg"`` or ``"application/pdf"``.
    name:
        Optional filename hint used for display / debugging.

    Examples
    --------
    >>> # From a file path
    >>> img = RactoFile.from_path("/tmp/photo.jpg")

    >>> # From bytes
    >>> img = RactoFile.from_bytes(open("photo.jpg", "rb").read(), "image/jpeg")
    """

    def __init__(self, data: bytes, mime_type: str, name: str = "") -> None:
        self.data = data
        self.mime_type = mime_type
        self.name = name

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: str | Path) -> RactoFile:
        """Load a file from *path* and auto-detect its MIME type.

        Parameters
        ----------
        path:
            Absolute or relative path to the file on disk.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        p = Path(path)
        mime_type, _ = mimetypes.guess_type(str(p))
        if mime_type is None:
            mime_type = "application/octet-stream"
        return cls(data=p.read_bytes(), mime_type=mime_type, name=p.name)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str, name: str = "") -> RactoFile:
        """Create a :class:`RactoFile` directly from *data* bytes.

        Parameters
        ----------
        data:
            Raw file bytes.
        mime_type:
            MIME type of the data, e.g. ``"image/png"``.
        name:
            Optional filename string (no file I/O is performed).
        """
        return cls(data=data, mime_type=mime_type, name=name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base64_data(self) -> str:
        """Return file bytes encoded as a base-64 ASCII string."""
        return base64.b64encode(self.data).decode("ascii")

    @property
    def is_image(self) -> bool:
        """True when the MIME type is a supported image type."""
        return self.mime_type in _IMAGE_MIMES

    @property
    def is_pdf(self) -> bool:
        return self.mime_type == "application/pdf"

    @property
    def is_text(self) -> bool:
        return self.mime_type.startswith("text/")

    def __repr__(self) -> str:
        label = self.name or "<unnamed>"
        return f"RactoFile(name={label!r}, mime_type={self.mime_type!r}, bytes={len(self.data)})"


# ---------------------------------------------------------------------------
# Provider-specific content-block builders
# ---------------------------------------------------------------------------


def _build_openai_content(
    user_message: str,
    attachments: list[RactoFile],
) -> str | list[dict[str, Any]]:
    """Return OpenAI-compatible user content with optional file attachments.

    Images become ``image_url`` blocks using an inline ``data:`` URI.
    Text files are embedded as ``text`` blocks.
    All other file types are embedded as a ``data:`` URI so vision-capable
    models can still attempt to process them.
    """
    if not attachments:
        return user_message

    parts: list[dict[str, Any]] = []
    for f in attachments:
        if f.is_image or not (f.is_text or f.is_pdf):
            # Images *and* any binary non-text file → data URI image_url block.
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{f.mime_type};base64,{f.base64_data}"},
                }
            )
        else:
            # Plain-text / unknown text: decode and embed as a text block.
            parts.append(
                {
                    "type": "text",
                    "text": f.data.decode("utf-8", errors="replace"),
                }
            )
    parts.append({"type": "text", "text": user_message})
    return parts


def _build_anthropic_content(
    user_message: str,
    attachments: list[RactoFile],
) -> str | list[dict[str, Any]]:
    """Return Anthropic-compatible user content with optional file attachments.

    * Images  → ``image`` content blocks (base-64 source).
    * PDFs    → ``document`` content blocks (base-64 source).
    * Text    → ``text`` content blocks (decoded string).
    * Other   → ``text`` block containing the base-64 payload with a label.
    """
    if not attachments:
        return user_message

    parts: list[dict[str, Any]] = []
    for f in attachments:
        if f.is_image:
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f.mime_type,
                        "data": f.base64_data,
                    },
                }
            )
        elif f.is_pdf:
            parts.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": f.base64_data,
                    },
                }
            )
        elif f.is_text:
            parts.append({"type": "text", "text": f.data.decode("utf-8", errors="replace")})
        else:
            label = f.name or "attachment"
            parts.append(
                {
                    "type": "text",
                    "text": (f"[File: {label} ({f.mime_type}) — base64 encoded]\n{f.base64_data}"),
                }
            )
    parts.append({"type": "text", "text": user_message})
    return parts


def _build_google_content(
    user_message: str,
    attachments: list[RactoFile],
) -> str | list[dict[str, Any]]:
    """Return Google Gemini-compatible user content with optional file attachments.

    Text files become ``text`` parts; all other files become ``inline_data``
    parts with base-64 encoded bytes and their MIME type.
    """
    if not attachments:
        return user_message

    parts: list[dict[str, Any]] = []
    for f in attachments:
        if f.is_text:
            parts.append({"text": f.data.decode("utf-8", errors="replace")})
        else:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": f.mime_type,
                        "data": f.base64_data,
                    }
                }
            )
    parts.append({"text": user_message})
    return parts


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
            "Optional few-shot examples. Each dict should have 'input' and 'output' keys."
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
    def _validate_constraints_not_empty_strings(self) -> RactoPrompt:
        for idx, c in enumerate(self.constraints):
            if not c.strip():
                raise ValueError(
                    f"constraints[{idx}] is blank. Every constraint must be a non-empty string."
                )
        return self

    @model_validator(mode="after")
    def _validate_examples_shape(self) -> RactoPrompt:
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
        constraint_lines = "\n".join(f"- {c}" for c in self.constraints)
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
                    f"Example {i}:\n  Input:  {ex['input']}\n  Output: {ex['output']}"
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
        attachments: list[RactoFile] | None = None,
        provider: str = "generic",
    ) -> list[dict[str, Any]]:
        """Return a ready-to-send message list for a given LLM provider.

        Parameters
        ----------
        user_message:
            The end-user's query or input.
        attachments:
            Optional list of :class:`RactoFile` objects to send alongside
            the text message.  Accepted inputs per file:

            * **File path** — use :meth:`RactoFile.from_path`::

                RactoFile.from_path("/tmp/diagram.png")

            * **Raw bytes** — use :meth:`RactoFile.from_bytes`::

                RactoFile.from_bytes(img_bytes, "image/png")

            Each file is re-encoded into the content-block schema expected
            by the target provider (``image_url`` for OpenAI, ``image`` /
            ``document`` for Anthropic, ``inline_data`` for Google).
        provider:
            One of ``"openai"``, ``"anthropic"``, ``"google"``, or
            ``"generic"``.  Controls the system-role key name and the
            content-block format used for attachments.

        Returns
        -------
        list[dict[str, Any]]
            A list of message dicts suitable for the provider's API.
        """
        system_prompt = self.compile()
        files = attachments or []

        if provider in ("openai", "generic"):
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_openai_content(user_message, files)},
            ]

        if provider == "anthropic":
            # Anthropic uses "system" as a top-level param, but for message
            # list representation we use the same structure — the adapter
            # will unpack it.
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_anthropic_content(user_message, files)},
            ]

        if provider == "google":
            # Gemini separates system_instruction from contents.
            # The adapter will split this; we use a marker role.
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_google_content(user_message, files)},
            ]

        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Expected one of: 'openai', 'anthropic', 'google', 'generic'."
        )

    def __str__(self) -> str:
        """Return the compiled prompt when cast to str."""
        return self.compile()
