# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.2] - 2026-03-07

### Added

- **AgentPipeline — 5 production upgrades** (`ractogateway.pipelines.agent`):
  - **Parallel tool execution**: LLM can now emit `{"tool_calls": [...]}` to run multiple independent tools simultaneously. Sync path uses `ThreadPoolExecutor` (controlled by `max_parallel_tools`); async path uses `asyncio.gather`.
  - **State-aware memory**: Each transcript now includes a "PRIOR TOOL RESULTS" memo so the LLM avoids redundant calls. Deduplication key is `tool_name(sorted_json_input)`.
  - **Graceful retries**: New `tool_retries` constructor param passed to `ToolExecutor` — failing tools are retried transparently before the error is reported to the LLM.
  - **Dynamic step scaling**: New built-in `request_more_steps(additional, reason)` tool; loop uses a mutable `step_cap`; capped by `max_step_extension` (default `0`, opt-in).
  - **Structured output validation**: `run(response_format=MyModel)` / `arun(response_format=MyModel)` parse the final answer into a Pydantic model stored in `AgentResult.parsed_output`; falls back to a single LLM correction call on parse failure.
- **AgentPipeline — per-stage error isolation** for `VideoProcessorPipeline` (`StageError` model with `stage`, `error_type`, `message`, `traceback`; `result.stage_errors` list; `result.has_errors` / `result.is_failed` properties).
- **Promptless kit usage**: All five developer kits (`OpenAI`, `Google`, `Anthropic`, `HuggingFace`, `Ollama`) now work without a `RactoPrompt`. Calling `kit.chat(ChatConfig(user_message=...))` without a prompt or `default_prompt` automatically applies a sensible built-in assistant prompt instead of raising `ValueError`.

### Changed

- `AgentPipeline._parse_response` return type changed from `tuple[str | None, str, dict]` to `tuple[str | None, list[tuple[str, dict]]]` to support parallel calls.
- `AgentPipeline._run_loop` / `_arun_loop` converted from `for` to `while` loops with mutable `step_cap` to enable dynamic step extension.
- Helper logic extracted into `_filter_calls`, `_exec_sync`, `_exec_async`, `_apply_results` private methods for readability and branch-count compliance.
- `_build_system_prompt` extra-rules injection renumbered to rule 9 (was 7).
- Version bumped to `0.2.2`.

### Fixed

- `ValueError: No prompt in ChatConfig and no default_prompt on the kit` — kits now return a default prompt instead of raising, enabling bare `kit.chat(ChatConfig(user_message=...))` usage across all providers.
- `LLMResponse.content` (not `.text`) used correctly in `AgentPipeline` loop — fixes `AttributeError` that prevented the agent from running.
- `_build_system_prompt` `extra_rules` numbering corrected (rule 9, not 7).

---

## [0.2.1] - 2026-03-05

### Added

- **PageIndexRAG (vectorless RAG)** in `ractogateway.rag.page_index`: BM25 ranking + keyword decision index for page-level retrieval without embeddings or a vector database.
- **Native thinking support** across provider adapters via `ChatConfig.native_thinking` and `ChatConfig.thinking_budget`, including reasoning deltas in streaming (`StreamDelta.thinking`, `StreamChunk.accumulated_thinking`) and `LLMResponse.thinking` for supported providers.
- **VideoProcessor pipeline** in `ractogateway.pipelines.video_processor` with frame extraction/deduplication, transcription backends, frame analysis, summarization, optional RAG storage, and sync/async APIs.
- **Agent pipeline** in `ractogateway.pipelines.agent` with ReAct loop, pluggable tools, built-in `finish`/RAG/SQL/HTTP/memory tool factories, and sync/async execution.
- **Test coverage** for the new pipelines: `tests/test_video_processor.py` and `tests/test_agent_pipeline.py`.
- API and user-guide docs for PageIndexRAG, native thinking, video pipeline, and agent pipeline.

### Changed

- Export surface updated to expose new pipelines and RAG modules from `ractogateway.pipelines` and `ractogateway.rag`.
- Package version source remains centralized in `src/ractogateway/_version.py`, now set to `0.2.1`.
- Documentation and API reference navigation expanded for new pipeline and RAG pages.

### Fixed

- `GatewayMetricsMiddleware` metric description text updated for clearer Prometheus metric semantics.
- Formatting corrections in PageIndexRAG documentation pages.

---

## [0.1.4] — 2026-02-24

### Fixed

- **mypy strict** — `history_turns` in all three developer kits (`OpenAIDeveloperKit`,
  `GoogleDeveloperKit`, `AnthropicDeveloperKit`) now correctly typed as
  `list[ChatTurn] | None` instead of `list[dict[str, str]] | None`, eliminating
  18 `[arg-type]` errors. `m.role.value` is used in place of `m.role` so the
  string literal matches `ChatTurn.role: Literal["system","user","assistant"]`.
- Removed stale `# type: ignore[return-value]` comment from `adapters/base.py`
  that mypy flagged as `[unused-ignore]`.
- Removed redundant `# noqa: PLC0415` from `adapters/google_kit.py` (rule
  already globally suppressed in `pyproject.toml`).

### Changed

- Version bumped from `0.1.3` to `0.1.4` in `pyproject.toml`, `__init__.py`,
  `docs/conf.py`, and `README.md` badge.
- User guide (`docs/guide/userguide.md`) encoding artifacts corrected:
  em-dashes (`—`) and degree symbol (`°`) in the Tool Calling section restored
  from corrupted `?` characters; output code block re-labelled `text` instead
  of `json` to fix Sphinx highlighting warning.

---

## [0.1.3] — 2026-02-22

### Added

- **OpenAI Structured Outputs support** (`ractogateway.adapters._openai_schema`):
  - `sanitize_for_openai(schema)` — recursively strips all keywords rejected by the
    OpenAI `json_schema` response format (`default`, `title`, `minimum`, `maximum`,
    `minLength`, `maxLength`, `pattern`, `format`, `minItems`, `maxItems`, etc.) and
    enforces strict-mode invariants (`additionalProperties: false`, all properties
    listed in `required`) throughout the schema tree including `$defs`.
  - `validate_schema_for_openai(schema, model_name)` — early sanity check that raises
    a descriptive `ValueError` for constructs that cannot be auto-fixed (`not`,
    `if/then/else`, `allOf`, unsupported primitive types) *before* any API call is made.
  - `build_response_format(model)` — validates, sanitises, and returns the
    `{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}`
    dict ready to pass as `response_format` to OpenAI Chat Completions.
- **41 unit tests** in `tests/test_openai_schema.py` covering keyword stripping,
  strict-mode enforcement, `Optional` field handling, nested models, early validation
  errors, and `build_response_format` output.

### Changed

- `OpenAILLMKit._build_request` now automatically sets `response_format` via
  `build_response_format` whenever `prompt.output_format` is a Pydantic `BaseModel`
  subclass. Users can override by passing `response_format=...` in `kwargs`.
- `_schema_from_model` in `ractogateway.prompts.engine` now strips the same
  Pydantic-generated noise keywords from prompt-embedded schemas (previously only
  `title` was removed).
- Version bumped from `0.1.2` to `0.1.3` in `pyproject.toml`, `__init__.py`, and
  `docs/conf.py` (docs was incorrectly pinned to `0.1.1`).

---

## [0.1.1] — 2026-02-19

### Added

- **RAG pipeline** — full `ractogateway.rag` subsystem:
  - File readers: `TextReader`, `PdfReader`, `WordReader`, `SpreadsheetReader`, `ImageReader`, `HtmlReader`, `FileReaderRegistry`
  - Chunkers: `FixedChunker`, `RecursiveChunker`, `SentenceChunker`, `SemanticChunker`
  - Text processors: `TextCleaner`, `Lemmatizer`, `ProcessingPipeline`
  - Embedders: `OpenAIEmbedder`, `GoogleEmbedder`, `VoyageEmbedder`
  - Vector stores: `InMemoryVectorStore`, `ChromaStore`, `FAISSStore`, `PineconeStore`, `QdrantStore`, `WeaviateStore`, `MilvusStore`, `PGVectorStore`
  - `RactoRAG` main pipeline class
  - RAG install extras: `[rag]`, `[rag-all]`, `[rag-pdf]`, `[rag-word]`, `[rag-excel]`, `[rag-image]`, `[rag-nlp]`, `[rag-chroma]`, `[rag-faiss]`, `[rag-pinecone]`, `[rag-qdrant]`, `[rag-weaviate]`, `[rag-milvus]`, `[rag-pgvector]`, `[rag-voyage]`
- **Fine-tuning pipeline** — `ractogateway.finetune` subsystem:
  - `RactoDataset` with validation and JSONL export
  - `OpenAIFineTuner` — dataset upload and job management
  - `GeminiFineTuner` — single-turn text-pair fine-tuning
  - `AnthropicFineTuner` stub
- **`RactoFile`** — file attachment support in `RactoPrompt` for multimodal inputs
- **Short aliases** for `Chat` classes in each developer kit (`__init__.py` re-exports)
- **CI/CD workflows** — `python-app.yml` (tests) and `python-publish.yml` (PyPI publish)

### Changed

- Version bumped from `0.1.0` to `0.1.1` in `pyproject.toml` and `src/ractogateway/__init__.py`
- Developer kit `__init__.py` files updated with usage examples and cleaner public API surface
- Refactored top-level imports across all kits for consistency
- Updated linting rules in `pyproject.toml` (`PLC0415`, `PLR0913` ignores formalised)

---

## [0.1.0] — 2026-02-18

### Added

- Initial release of **RactoGateway** — a unified, production-ready AI SDK
- **RACTO Prompt Engine** (`ractogateway.prompts.engine`):
  - `RactoPrompt` — structured prompt with role, task, context, constraints, and `output_format`
  - Anti-hallucination guardrails built into every prompt
- **Developer Kits** with sync and async parity:
  - `OpenAIDeveloperKit` — chat, streaming, embeddings, tool calling
  - `GoogleDeveloperKit` — chat, streaming, embeddings, tool calling
  - `AnthropicDeveloperKit` — chat, streaming, tool calling
- **Adapter layer** (`ractogateway.adapters`):
  - `BaseLLMAdapter` abstract base class
  - `OpenAILLMKit`, `GoogleLLMKit`, `AnthropicLLMKit` provider adapters
- **Tool Registry** (`ractogateway.tools.registry`):
  - `@tool` decorator for registering callable tools
  - `ToolRegistry` for managing and passing tools to LLMs
- **Gateway runner** (`ractogateway.gateway`) — low-level provider-agnostic execution layer
- **Pydantic models** (`ractogateway._models`): `ChatResponse`, `EmbeddingResponse`, `StreamChunk`
- `py.typed` marker — full PEP 561 type information
- Install extras: `[openai]`, `[google]`, `[anthropic]`, `[all]`
- Apache 2.0 license

---

[0.2.1]: https://github.com/IAMPathak2702/RactoGateway/compare/v0.1.4...v0.2.1
[0.1.4]: https://github.com/IAMPathak2702/RactoGateway/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/IAMPathak2702/RactoGateway/compare/v0.1.1...v0.1.3
[0.1.1]: https://github.com/IAMPathak2702/RactoGateway/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/IAMPathak2702/RactoGateway/releases/tag/v0.1.0
