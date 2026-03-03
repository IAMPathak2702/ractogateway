"""Strongly-typed input models for chat completion calls."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ractogateway.prompts.engine import RactoFile, RactoPrompt
from ractogateway.tools.registry import ToolRegistry


class MessageRole(str, Enum):
    """Role of a single message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single conversation turn.

    Used inside ``ChatConfig.history`` to provide prior conversation context
    to the model for multi-turn conversations.
    """

    role: MessageRole
    content: str


class ChatConfig(BaseModel):
    """Validated input for every ``chat`` / ``achat`` / ``stream`` / ``astream`` call.

    Pass a single ``ChatConfig`` to any developer-kit method.  Every field has
    a safe default so you only need to supply what you actually need.

    Minimal example::

        config = ChatConfig(user_message="Explain Python generators.")
        response = kit.chat(config)

    Vision / multimodal example::

        from ractogateway.prompts.engine import RactoFile

        config = ChatConfig(
            user_message="Describe this chart.",
            attachments=[RactoFile.from_path("sales_q4.png")],
        )

    Structured JSON output example::

        class Sentiment(BaseModel):
            label: str
            score: float

        config = ChatConfig(
            user_message="I love this library!",
            response_model=Sentiment,
        )
    """

    user_message: str = Field(
        ...,
        min_length=1,
        description=(
            "The end-user's text query or instruction sent to the model. "
            "When attachments are also provided this text is always included "
            "as the final 'text' part of the user turn so the model receives "
            "both the files and your question in one message. "
            "Minimum length: 1 character."
        ),
    )
    prompt: RactoPrompt | None = Field(
        default=None,
        description=(
            "Structured RACTO prompt that compiles to the system instruction. "
            "When ``None`` the kit's ``default_prompt`` is used instead; "
            "at least one of the two must be set or ``chat()`` will raise. "
            "The prompt defines the model's role, aim, constraints, tone, and "
            "expected output format — including JSON schemas for structured output."
        ),
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature controlling output randomness. "
            "0.0 = fully deterministic / greedy decoding (best for structured "
            "JSON, code generation, and factual Q&A). "
            "0.3-0.7 = balanced creativity (good for summaries, analysis). "
            "0.8-1.2 = creative writing, brainstorming. "
            "Above 1.2 = highly random; use with caution. "
            "Clamped to [0.0, 2.0] by validation."
        ),
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description=(
            "Maximum number of tokens the model may generate in its response. "
            "Does not count the prompt / input tokens. "
            "Increase for long-form documents or multi-step reasoning chains; "
            "decrease to enforce concise replies or reduce cost. "
            "Must be greater than 0."
        ),
    )
    tools: ToolRegistry | None = Field(
        default=None,
        description=(
            "Optional ``ToolRegistry`` containing Python functions the model "
            "may call (function / tool calling). "
            "Create one with ``ToolRegistry([fn1, fn2])`` where each function "
            "is decorated with ``@tool``. "
            "When ``None`` tool calling is disabled. "
            "Combine with ``auto_execute_tools=True`` to let the kit run the "
            "functions automatically and continue the loop."
        ),
    )
    auto_execute_tools: bool = Field(
        default=False,
        description=(
            "When ``True``, ``chat()`` / ``achat()`` automatically execute every "
            "tool call requested by the model, feed the results back, and repeat "
            "until the model produces a final text response or ``max_tool_turns`` "
            "is reached. "
            "When ``False`` (default) tool-call results are returned in "
            "``LLMResponse.tool_calls`` for you to handle manually. "
            "Requires ``tools`` to be set."
        ),
    )
    max_tool_turns: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Maximum number of automatic tool-execution rounds before the kit "
            "returns the last response as-is. "
            "Only meaningful when ``auto_execute_tools=True``. "
            "Range: 1-10."
        ),
    )
    response_model: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Optional Pydantic model class for structured JSON output. "
            "When set the kit instructs the model to respond with a JSON object "
            "that matches the model's schema, then validates and parses it. "
            "The validated dict is available as ``LLMResponse.parsed``. "
            "Combine with ``max_validation_retries`` to auto-correct bad JSON. "
            "Example: ``response_model=MySentimentModel``."
        ),
    )
    max_validation_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Number of automatic retry attempts when ``response_model`` "
            "validation fails. "
            "On each retry the exact Pydantic validation errors together with "
            "the invalid JSON are fed back to the model as a correction prompt "
            "so it can fix its output. "
            "Set to 0 to disable retries and raise "
            "``ResponseModelValidationError`` immediately on the first failure. "
            "Range: 0-5."
        ),
    )
    history: list[Message] = Field(
        default_factory=list,
        description=(
            "Prior conversation turns for multi-turn / stateful chat. "
            "Each entry is a ``Message(role=MessageRole.USER/ASSISTANT, "
            "content='...')`` object. "
            "Turns are spliced between the system message and the current user "
            "message before the API call so the model sees the full context. "
            "Defaults to an empty list (single-turn / stateless mode)."
        ),
    )
    attachments: list[RactoFile] | None = Field(
        default=None,
        description=(
            "Optional list of file attachments for multimodal / VLM calls. "
            "Create files with ``RactoFile.from_path('image.jpg')`` (auto-detects "
            "MIME type) or ``RactoFile.from_bytes(raw_bytes, 'image/png')``. "
            "\n\n"
            "Attachment processing differs per provider and file type:\n\n"
            "**OpenAI / HuggingFace** (``image_url`` content blocks):\n"
            "  - Images (JPEG, PNG, GIF, WebP) → inline ``data:`` URI "
            "``image_url`` block; processed by GPT-4o, GPT-4-vision, "
            "LLaVA-HF, Qwen-VL, etc.\n"
            "  - Text files (CSV, TXT, HTML, Markdown) → ``text`` block; "
            "the raw UTF-8 content is embedded and the model reads it.\n"
            "  - PDFs → ``text`` block (raw bytes decoded); for reliable PDF "
            "understanding prefer the Anthropic kit which has native PDF support.\n"
            "  - XLSX / DOCX / other binary → ``image_url`` data-URI fallback; "
            "the model will not meaningfully parse binary spreadsheets — "
            "pre-convert to CSV or text before attaching.\n\n"
            "**Anthropic** (native content blocks):\n"
            "  - Images → ``image`` block with base-64 source; "
            "Claude 3.x and later process these natively.\n"
            "  - PDFs → ``document`` block with base-64 source; "
            "Claude has first-class PDF understanding (text + layout).\n"
            "  - Text (CSV, TXT, Markdown) → ``text`` block (UTF-8).\n"
            "  - XLSX / other binary → ``text`` block showing "
            "``[File: name (mime_type) — base64 encoded]`` followed by the "
            "base-64 payload; the model sees the raw encoding, not parsed data. "
            "Pre-convert spreadsheets to CSV.\n\n"
            "**Google Gemini** (``inline_data`` parts):\n"
            "  - Text (CSV, TXT) → ``{\"text\": decoded_content}`` part.\n"
            "  - All other files (images, PDFs, XLSX, audio, video) → "
            "``{\"inline_data\": {\"mime_type\": ..., \"data\": base64}}`` part; "
            "Gemini 2.x supports images, PDFs, audio (MP3/WAV), video (MP4), "
            "and spreadsheets (XLSX) natively.\n\n"
            "**Ollama** (``images`` list on the message):\n"
            "  - Images → placed in the top-level ``images`` list "
            "(base-64 strings) understood by llava, llava-llama3, minicpm-v, "
            "bakllava, and other Ollama vision models.\n"
            "  - Text files (CSV, TXT) → prepended as UTF-8 text in the "
            "``content`` field before your ``user_message``.\n"
            "  - PDFs / XLSX / other binary → silently ignored by the Ollama "
            "adapter (Ollama has no native document support). Pre-extract text.\n\n"
            "**Recommendation by file type**:\n"
            "  - Images → all 5 kits work; Ollama requires a vision model.\n"
            "  - PDFs → use Anthropic (native) or Gemini (inline_data).\n"
            "  - CSV / plain-text → all 5 kits embed as readable text.\n"
            "  - XLSX / DOCX → convert to CSV / Markdown first for best results.\n"
            "  - Audio / Video → Gemini only (inline_data with correct MIME)."
        ),
    )
    chain_of_thought: bool = Field(
        default=False,
        description=(
            "When ``True``, instructs the model to reason step by step before "
            "delivering its final answer. "
            "A chain-of-thought constraint is appended to the compiled system "
            "prompt, asking the model to state each reasoning step clearly and "
            "then conclude with its answer. "
            "Compatible with all five provider kits (OpenAI, Anthropic, Google, "
            "Ollama, HuggingFace). "
            "Combine with ``temperature > 0`` for richer reasoning traces."
        ),
    )
    native_thinking: bool = Field(
        default=False,
        description=(
            "Enable native model-level extended thinking / reasoning output. "
            "The model's actual reasoning process is returned alongside the answer. "
            "\n\n"
            "**Anthropic** (``claude-3-7-sonnet-20250219`` and later): "
            "Adds a ``thinking`` content block before the answer; temperature is "
            "automatically forced to 1 as required by the API. "
            "Thinking text streams in ``StreamChunk.delta.thinking`` / "
            "``StreamChunk.accumulated_thinking``; non-streaming responses expose "
            "it in ``LLMResponse.thinking``. "
            "\n\n"
            "**Google** (``gemini-2.5-pro``, ``gemini-2.0-flash-thinking-exp``): "
            "Thought parts are separated from the answer and returned in the same "
            "``thinking`` fields. "
            "\n\n"
            "**OpenAI** (``o1``, ``o3``, ``o3-mini``): Reasoning happens internally; "
            "no reasoning text is exposed but ``LLMResponse.usage[reasoning_tokens]`` "
            "shows how many tokens were consumed. "
            "\n\n"
            "Use ``thinking_budget`` to control reasoning token spend."
        ),
    )
    thinking_budget: int = Field(
        default=10000,
        gt=0,
        description=(
            "Maximum tokens the model may spend on internal thinking / reasoning. "
            "Only meaningful when ``native_thinking=True``. "
            "Higher budgets allow deeper reasoning but increase cost and latency. "
            "Anthropic: 1024-100000 (recommended >= 1024). "
            "Google: set to a positive integer; the model respects it as a soft cap. "
            "Must be greater than 0."
        ),
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific keyword arguments forwarded verbatim to the "
            "underlying API call. "
            "Common examples: "
            "``top_p`` (nucleus sampling probability), "
            "``stop`` (list of stop sequences), "
            "``seed`` (for reproducible sampling on OpenAI), "
            "``frequency_penalty`` / ``presence_penalty`` (OpenAI), "
            "``repetition_penalty`` (HuggingFace), "
            "``top_k`` (Ollama / HuggingFace). "
            "Unknown keys are silently forwarded so provider-specific features "
            "are always accessible without needing a new field."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}
