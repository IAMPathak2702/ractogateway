# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/IAMPathak2702/RactoGateway/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/IAMPathak2702/RactoGateway/releases/tag/v0.1.0
