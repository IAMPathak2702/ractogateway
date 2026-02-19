"""Training dataset primitives for multimodal LLM fine-tuning.

Classes
-------
RactoTrainingMessage
    One turn in a training conversation (role + text + optional file attachments).
RactoTrainingExample
    A complete multi-turn conversation used as a single training record.
RactoDataset
    Ordered collection of examples with validation, splitting, and JSONL export.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ractogateway.prompts.engine import RactoFile

RoleType = Literal["system", "user", "assistant"]
_SINGLE_TURN_MESSAGE_COUNT = 2


# ---------------------------------------------------------------------------
# RactoTrainingMessage
# ---------------------------------------------------------------------------


@dataclass
class RactoTrainingMessage:
    """One conversational turn inside a training example.

    Parameters
    ----------
    role : {"system", "user", "assistant"}
        Speaker role.
    content : str
        Text content of the message.
    attachments : list[RactoFile]
        Optional images / PDFs for multimodal training examples.
        Use :meth:`RactoFile.from_path` or :meth:`RactoFile.from_bytes`.
    """

    role: RoleType
    content: str
    attachments: list[RactoFile] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation helpers (one per provider)
    # ------------------------------------------------------------------

    def to_openai(self) -> dict[str, Any]:
        """Return an OpenAI-compatible message dict.

        Text-only messages produce ``{"role": ..., "content": str}``.
        Messages with attachments produce a content-block list:
        ``{"role": ..., "content": [image_url_block, ..., text_block]}``.
        """
        if not self.attachments:
            return {"role": self.role, "content": self.content}

        parts: list[dict[str, Any]] = []
        for f in self.attachments:
            if f.is_image or not (f.is_text or f.is_pdf):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{f.mime_type};base64,{f.base64_data}"},
                    }
                )
            else:
                parts.append({"type": "text", "text": f.data.decode("utf-8", errors="replace")})
        parts.append({"type": "text", "text": self.content})
        return {"role": self.role, "content": parts}

    def to_anthropic(self) -> dict[str, Any]:
        """Return an Anthropic-compatible message dict.

        System messages should be lifted to the top-level ``system`` field —
        :meth:`RactoTrainingExample.to_anthropic_dict` handles this automatically.
        """
        if not self.attachments:
            return {"role": self.role, "content": self.content}

        parts: list[dict[str, Any]] = []
        for f in self.attachments:
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
                        "text": f"[File: {label} ({f.mime_type})]\n{f.base64_data}",
                    }
                )
        parts.append({"type": "text", "text": self.content})
        return {"role": self.role, "content": parts}

    def to_gemini_parts(self) -> list[dict[str, Any]]:
        """Return a list of Gemini content parts (``text`` + ``inline_data``)."""
        parts: list[dict[str, Any]] = []
        for f in self.attachments:
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
        parts.append({"text": self.content})
        return parts


# ---------------------------------------------------------------------------
# RactoTrainingExample
# ---------------------------------------------------------------------------


class RactoTrainingExample:
    """A complete conversation used as one training record.

    Parameters
    ----------
    messages : list[RactoTrainingMessage]
        Ordered turns.  Typical shapes:

        * Single-turn  : ``[user, assistant]``
        * With system  : ``[system, user, assistant]``
        * Multi-turn   : ``[system, user, assistant, user, assistant, …]``

    Examples
    --------
    >>> ex = RactoTrainingExample.from_pair(
    ...     user="What is 2 + 2?",
    ...     assistant="4",
    ...     system="You are a maths tutor.",
    ... )

    >>> # Multimodal example (image + question)
    >>> ex = RactoTrainingExample.from_pair(
    ...     user="Describe this chart.",
    ...     assistant="The chart shows monthly revenue for Q4 2024.",
    ...     user_attachments=[RactoFile.from_path("chart.png")],
    ... )
    """

    def __init__(self, messages: list[RactoTrainingMessage]) -> None:
        self.messages = messages

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pair(
        cls,
        user: str,
        assistant: str,
        *,
        system: str = "",
        user_attachments: list[RactoFile] | None = None,
    ) -> RactoTrainingExample:
        """Create a single-turn (prompt → completion) training example.

        Parameters
        ----------
        user : str
            The user prompt.
        assistant : str
            The desired model response.
        system : str
            Optional system prompt prepended to the conversation.
        user_attachments : list[RactoFile] | None
            Images or other files attached to the user turn.
        """
        msgs: list[RactoTrainingMessage] = []
        if system:
            msgs.append(RactoTrainingMessage(role="system", content=system))
        msgs.append(
            RactoTrainingMessage(
                role="user",
                content=user,
                attachments=user_attachments or [],
            )
        )
        msgs.append(RactoTrainingMessage(role="assistant", content=assistant))
        return cls(msgs)

    @classmethod
    def from_conversation(
        cls,
        turns: list[tuple[RoleType, str]],
    ) -> RactoTrainingExample:
        """Build from a list of ``(role, content)`` tuples.

        Parameters
        ----------
        turns : list[tuple[str, str]]
            E.g. ``[("system", "…"), ("user", "…"), ("assistant", "…")]``
        """
        return cls([RactoTrainingMessage(role=r, content=c) for r, c in turns])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_openai_dict(self) -> dict[str, Any]:
        """Serialize to OpenAI fine-tuning JSONL record.

        Output format::

            {"messages": [{"role": "system", "content": "…"}, …]}
        """
        return {"messages": [m.to_openai() for m in self.messages]}

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Serialize to Anthropic fine-tuning JSONL record.

        Output format::

            {"system": "…", "messages": [{"role": "user", …}, …]}

        The ``system`` key is only present when a system message exists.
        """
        system = ""
        messages: list[dict[str, Any]] = []
        for m in self.messages:
            if m.role == "system":
                system = m.content
            else:
                messages.append(m.to_anthropic())
        record: dict[str, Any] = {"messages": messages}
        if system:
            record["system"] = system
        return record

    def to_gemini_dict(self) -> dict[str, Any]:
        """Serialize to Gemini tuning record.

        For text-only single-turn examples (most common) the output is::

            {"text_input": "…", "output": "…"}

        For multimodal or multi-turn examples the Vertex AI ``contents``
        format is used::

            {"contents": [{"role": "user", "parts": […]}, …]}
        """
        has_attachments = any(m.attachments for m in self.messages)
        non_system = [m for m in self.messages if m.role != "system"]

        if not has_attachments and len(non_system) == _SINGLE_TURN_MESSAGE_COUNT:
            user_msg = next(m for m in non_system if m.role == "user")
            asst_msg = next(m for m in non_system if m.role == "assistant")
            return {"text_input": user_msg.content, "output": asst_msg.content}

        # Multi-turn or multimodal → Vertex AI contents format
        contents: list[dict[str, Any]] = []
        for m in non_system:
            gemini_role = "model" if m.role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": m.to_gemini_parts()})
        return {"contents": contents}

    def __repr__(self) -> str:
        return f"RactoTrainingExample(turns={len(self.messages)})"


# ---------------------------------------------------------------------------
# RactoDataset
# ---------------------------------------------------------------------------


class RactoDataset:
    """An ordered collection of :class:`RactoTrainingExample` objects.

    This is the central data container for building, validating, splitting,
    and exporting fine-tuning datasets for any supported LLM provider.

    Parameters
    ----------
    examples : list[RactoTrainingExample] | None
        Initial examples.  An empty dataset is created when omitted.

    Examples
    --------
    Build from (user, assistant) pairs::

        ds = RactoDataset.from_pairs(
            [
                ("What is Python?", "Python is a high-level programming language."),
                ("What is a list?", "A list is a mutable ordered sequence."),
            ],
            system="You are a Python tutor.",
        )

    Add multimodal examples manually::

        ds.add(
            RactoTrainingExample.from_pair(
                user="Describe this image.",
                assistant="The image shows a flowchart with three decision nodes.",
                user_attachments=[RactoFile.from_path("diagram.png")],
            )
        )

    Export to JSONL for fine-tuning::

        train_ds, val_ds = ds.split(0.8, seed=42)
        train_ds.export_jsonl("train.jsonl", provider="openai")
        val_ds.export_jsonl("val.jsonl", provider="openai")
    """

    def __init__(self, examples: list[RactoTrainingExample] | None = None) -> None:
        self._examples: list[RactoTrainingExample] = list(examples or [])

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def add(self, example: RactoTrainingExample) -> None:
        """Append a single training example."""
        self._examples.append(example)

    def extend(self, examples: list[RactoTrainingExample]) -> None:
        """Append multiple training examples at once."""
        self._examples.extend(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[RactoTrainingExample]:
        return iter(self._examples)

    def __getitem__(self, idx: int) -> RactoTrainingExample:
        return self._examples[idx]

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pairs(
        cls,
        pairs: list[tuple[str, str]],
        *,
        system: str = "",
    ) -> RactoDataset:
        """Build a text-only dataset from ``(user, assistant)`` pairs.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            Each tuple is ``(user_message, expected_assistant_response)``.
        system : str
            Optional system prompt applied uniformly to every example.
        """
        return cls([RactoTrainingExample.from_pair(u, a, system=system) for u, a in pairs])

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        provider: str = "openai",
    ) -> RactoDataset:
        """Load a JSONL dataset previously exported for *provider*.

        Supports text-only OpenAI, Anthropic, and Gemini formats.

        Parameters
        ----------
        path : str | Path
            Path to the ``.jsonl`` file.
        provider : str
            One of ``"openai"``, ``"anthropic"``, ``"gemini"``.
        """
        p = Path(path)
        examples: list[RactoTrainingExample] = []

        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)

            if provider in ("openai", "generic"):
                msgs = [
                    RactoTrainingMessage(
                        role=m["role"],
                        content=m["content"]
                        if isinstance(m["content"], str)
                        else str(m["content"]),
                    )
                    for m in record.get("messages", [])
                ]

            elif provider == "anthropic":
                msgs = []
                if "system" in record:
                    msgs.append(RactoTrainingMessage(role="system", content=record["system"]))
                msgs += [
                    RactoTrainingMessage(
                        role=m["role"],
                        content=m["content"]
                        if isinstance(m["content"], str)
                        else str(m["content"]),
                    )
                    for m in record.get("messages", [])
                ]

            elif provider == "gemini":
                if "text_input" in record:
                    msgs = [
                        RactoTrainingMessage(role="user", content=record["text_input"]),
                        RactoTrainingMessage(role="assistant", content=record["output"]),
                    ]
                else:
                    msgs = [
                        RactoTrainingMessage(
                            role="assistant" if c["role"] == "model" else c["role"],
                            content=c["parts"][0].get("text", "") if c.get("parts") else "",
                        )
                        for c in record.get("contents", [])
                    ]
            else:
                raise ValueError(
                    f"Unknown provider {provider!r}. Choose from: 'openai', 'anthropic', 'gemini'."
                )

            examples.append(RactoTrainingExample(msgs))

        return cls(examples)

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def shuffle(self, seed: int | None = None) -> RactoDataset:
        """Return a new dataset with examples in random order.

        Parameters
        ----------
        seed : int | None
            Optional random seed for reproducibility.
        """
        examples = list(self._examples)
        rng = random.Random(seed)  # noqa: S311 - deterministic shuffling is intentional.
        rng.shuffle(examples)
        return RactoDataset(examples)

    def split(
        self,
        train_ratio: float = 0.8,
        *,
        seed: int | None = None,
    ) -> tuple[RactoDataset, RactoDataset]:
        """Split into train and validation datasets.

        Parameters
        ----------
        train_ratio : float
            Fraction of examples for the training split.  Must be between
            0 and 1 (exclusive).
        seed : int | None
            Optional random seed for reproducible shuffling.

        Returns
        -------
        tuple[RactoDataset, RactoDataset]
            ``(train_dataset, validation_dataset)``
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be strictly between 0 and 1, got {train_ratio}")
        shuffled = self.shuffle(seed=seed)
        n = int(len(shuffled) * train_ratio)
        return (
            RactoDataset(shuffled._examples[:n]),
            RactoDataset(shuffled._examples[n:]),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, provider: str = "openai") -> list[str]:
        """Check examples for common formatting errors.

        Parameters
        ----------
        provider : str
            Provider to validate against (``"openai"``, ``"anthropic"``,
            or ``"gemini"``).

        Returns
        -------
        list[str]
            A list of human-readable error strings.  An empty list means
            the dataset is ready to use.
        """
        errors: list[str] = []

        if len(self) == 0:
            errors.append("Dataset is empty.")
            return errors

        for i, ex in enumerate(self._examples):
            non_system = [m for m in ex.messages if m.role != "system"]

            if not non_system:
                errors.append(f"[example {i}] No user/assistant messages found.")
                continue

            if provider in ("openai", "generic", "anthropic"):
                if not any(m.role == "user" for m in non_system):
                    errors.append(f"[example {i}] Missing 'user' message.")
                if not any(m.role == "assistant" for m in non_system):
                    errors.append(f"[example {i}] Missing 'assistant' message.")
                if non_system[-1].role != "assistant":
                    errors.append(
                        f"[example {i}] Last message must be 'assistant', "
                        f"got '{non_system[-1].role}'."
                    )
                for m in ex.messages:
                    if not m.content.strip() and not m.attachments:
                        errors.append(f"[example {i}] Empty content in '{m.role}' message.")

            elif provider == "gemini":
                if len(non_system) < _SINGLE_TURN_MESSAGE_COUNT:
                    errors.append(f"[example {i}] Gemini needs at least one user+assistant pair.")
                user_count = sum(1 for m in non_system if m.role == "user")
                asst_count = sum(1 for m in non_system if m.role == "assistant")
                if user_count != asst_count:
                    errors.append(
                        f"[example {i}] Gemini requires equal user/assistant turns, "
                        f"got {user_count} user / {asst_count} assistant."
                    )

        return errors

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_jsonl_string(self, provider: str = "openai") -> str:
        """Serialize all examples to a JSONL string for *provider*.

        Parameters
        ----------
        provider : str
            One of ``"openai"`` / ``"generic"``, ``"anthropic"``, ``"gemini"``.
        """
        lines: list[str] = []
        for ex in self._examples:
            if provider in ("openai", "generic"):
                lines.append(json.dumps(ex.to_openai_dict(), ensure_ascii=False))
            elif provider == "anthropic":
                lines.append(json.dumps(ex.to_anthropic_dict(), ensure_ascii=False))
            elif provider == "gemini":
                lines.append(json.dumps(ex.to_gemini_dict(), ensure_ascii=False))
            else:
                raise ValueError(
                    f"Unknown provider {provider!r}. "
                    f"Choose from: 'openai', 'anthropic', 'gemini', 'generic'."
                )
        return "\n".join(lines) + "\n"

    def export_jsonl(
        self,
        path: str | Path,
        provider: str = "openai",
        *,
        overwrite: bool = False,
    ) -> Path:
        """Write the dataset to a ``.jsonl`` file on disk.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        provider : str
            One of ``"openai"``, ``"anthropic"``, ``"gemini"``.
        overwrite : bool
            When ``False`` (default), raise :exc:`FileExistsError` if the
            file already exists.

        Returns
        -------
        Path
            The resolved path of the written file.
        """
        p = Path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} already exists. Pass overwrite=True to replace it.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_jsonl_string(provider), encoding="utf-8")
        return p

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return brief statistics about the dataset.

        Returns
        -------
        dict
            Keys: ``examples``, ``total_messages``, ``avg_turns_per_example``,
            ``multimodal_examples``.
        """
        total_messages = sum(len(ex.messages) for ex in self._examples)
        multimodal_count = sum(
            1 for ex in self._examples if any(m.attachments for m in ex.messages)
        )
        return {
            "examples": len(self._examples),
            "total_messages": total_messages,
            "avg_turns_per_example": round(total_messages / max(len(self._examples), 1), 2),
            "multimodal_examples": multimodal_count,
        }

    def __repr__(self) -> str:
        return f"RactoDataset(examples={len(self._examples)})"
