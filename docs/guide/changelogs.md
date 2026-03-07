# Changelog

## [Unreleased]

### Added

- **PageIndexRAG**: Added `add_document`, `add_texts`, and `search` aliases to `PageIndexRAG` for compatibility with documentation examples and `AgentPipeline`.
- **PageIndexRAG**: Added `text` and `content` property aliases to `PageIndexResult` and `PageEntry`.
- **PageIndexRAG**: Added `pages` and `results` property aliases to `PageIndexResponse`.
- **Docs**: Added "OCR Backends" section to `PageIndexRAG` API reference.

### Changed

- **Docs**: Updated `PageIndexRAG` quick start examples in docstrings and user guide to match the implementation.
- **Docs**: Removed unused `content_hash` field from `PageEntry` documentation in the user guide.