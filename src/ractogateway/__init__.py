"""RactoGateway - unified AI SDK with anti-hallucination prompting via RACTO.

The package surface stays import-friendly while deferring heavy imports until a
specific symbol is accessed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ractogateway import (
        anthropic_developer_kit,
        finetune,
        google_developer_kit,
        openai_developer_kit,
        rag,
    )
    from ractogateway.adapters.base import LLMResponse
    from ractogateway.finetune import (
        AnthropicFineTuner,
        GeminiFineTuner,
        OpenAIFineTuner,
        RactoDataset,
        RactoTrainingExample,
        RactoTrainingMessage,
    )
    from ractogateway.gateway.runner import Gateway
    from ractogateway.prompts.engine import RactoFile, RactoPrompt
    from ractogateway.rag.chunkers.fixed_chunker import FixedChunker
    from ractogateway.rag.chunkers.recursive_chunker import RecursiveChunker
    from ractogateway.rag.chunkers.semantic_chunker import SemanticChunker
    from ractogateway.rag.chunkers.sentence_chunker import SentenceChunker
    from ractogateway.rag.embedders.google_embedder import GoogleEmbedder
    from ractogateway.rag.embedders.openai_embedder import OpenAIEmbedder
    from ractogateway.rag.embedders.voyage_embedder import VoyageEmbedder
    from ractogateway.rag.pipeline import RactoRAG
    from ractogateway.rag.processors.cleaner import TextCleaner
    from ractogateway.rag.processors.lemmatizer import Lemmatizer
    from ractogateway.rag.processors.pipeline import ProcessingPipeline
    from ractogateway.rag.readers.registry import FileReaderRegistry
    from ractogateway.rag.stores.chroma_store import ChromaStore
    from ractogateway.rag.stores.faiss_store import FAISSStore
    from ractogateway.rag.stores.in_memory_store import InMemoryVectorStore
    from ractogateway.rag.stores.milvus_store import MilvusStore
    from ractogateway.rag.stores.pgvector_store import PGVectorStore
    from ractogateway.rag.stores.pinecone_store import PineconeStore
    from ractogateway.rag.stores.qdrant_store import QdrantStore
    from ractogateway.rag.stores.weaviate_store import WeaviateStore
    from ractogateway.tools.registry import ToolRegistry, tool

_MODULE_EXPORTS: dict[str, str] = {
    "anthropic_developer_kit": "ractogateway.anthropic_developer_kit",
    "finetune": "ractogateway.finetune",
    "google_developer_kit": "ractogateway.google_developer_kit",
    "openai_developer_kit": "ractogateway.openai_developer_kit",
    "rag": "ractogateway.rag",
}

_ATTR_EXPORTS: dict[str, tuple[str, str]] = {
    "AnthropicFineTuner": ("ractogateway.finetune", "AnthropicFineTuner"),
    "GeminiFineTuner": ("ractogateway.finetune", "GeminiFineTuner"),
    "OpenAIFineTuner": ("ractogateway.finetune", "OpenAIFineTuner"),
    "RactoDataset": ("ractogateway.finetune", "RactoDataset"),
    "RactoTrainingExample": ("ractogateway.finetune", "RactoTrainingExample"),
    "RactoTrainingMessage": ("ractogateway.finetune", "RactoTrainingMessage"),
    "Gateway": ("ractogateway.gateway.runner", "Gateway"),
    "LLMResponse": ("ractogateway.adapters.base", "LLMResponse"),
    "RactoFile": ("ractogateway.prompts.engine", "RactoFile"),
    "RactoPrompt": ("ractogateway.prompts.engine", "RactoPrompt"),
    "ToolRegistry": ("ractogateway.tools.registry", "ToolRegistry"),
    "tool": ("ractogateway.tools.registry", "tool"),
    "RactoRAG": ("ractogateway.rag.pipeline", "RactoRAG"),
    "FixedChunker": ("ractogateway.rag.chunkers.fixed_chunker", "FixedChunker"),
    "RecursiveChunker": ("ractogateway.rag.chunkers.recursive_chunker", "RecursiveChunker"),
    "SemanticChunker": ("ractogateway.rag.chunkers.semantic_chunker", "SemanticChunker"),
    "SentenceChunker": ("ractogateway.rag.chunkers.sentence_chunker", "SentenceChunker"),
    "GoogleEmbedder": ("ractogateway.rag.embedders.google_embedder", "GoogleEmbedder"),
    "OpenAIEmbedder": ("ractogateway.rag.embedders.openai_embedder", "OpenAIEmbedder"),
    "VoyageEmbedder": ("ractogateway.rag.embedders.voyage_embedder", "VoyageEmbedder"),
    "Lemmatizer": ("ractogateway.rag.processors.lemmatizer", "Lemmatizer"),
    "ProcessingPipeline": ("ractogateway.rag.processors.pipeline", "ProcessingPipeline"),
    "TextCleaner": ("ractogateway.rag.processors.cleaner", "TextCleaner"),
    "FileReaderRegistry": ("ractogateway.rag.readers.registry", "FileReaderRegistry"),
    "ChromaStore": ("ractogateway.rag.stores.chroma_store", "ChromaStore"),
    "FAISSStore": ("ractogateway.rag.stores.faiss_store", "FAISSStore"),
    "InMemoryVectorStore": ("ractogateway.rag.stores.in_memory_store", "InMemoryVectorStore"),
    "MilvusStore": ("ractogateway.rag.stores.milvus_store", "MilvusStore"),
    "PGVectorStore": ("ractogateway.rag.stores.pgvector_store", "PGVectorStore"),
    "PineconeStore": ("ractogateway.rag.stores.pinecone_store", "PineconeStore"),
    "QdrantStore": ("ractogateway.rag.stores.qdrant_store", "QdrantStore"),
    "WeaviateStore": ("ractogateway.rag.stores.weaviate_store", "WeaviateStore"),
}

__all__ = [
    "AnthropicFineTuner",
    "GeminiFineTuner",
    "OpenAIFineTuner",
    "RactoDataset",
    "RactoTrainingExample",
    "RactoTrainingMessage",
    "Gateway",
    "LLMResponse",
    "RactoFile",
    "RactoPrompt",
    "ToolRegistry",
    "tool",
    "RactoRAG",
    "FixedChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    "GoogleEmbedder",
    "OpenAIEmbedder",
    "VoyageEmbedder",
    "Lemmatizer",
    "ProcessingPipeline",
    "TextCleaner",
    "FileReaderRegistry",
    "ChromaStore",
    "FAISSStore",
    "InMemoryVectorStore",
    "MilvusStore",
    "PGVectorStore",
    "PineconeStore",
    "QdrantStore",
    "WeaviateStore",
    "anthropic_developer_kit",
    "finetune",
    "google_developer_kit",
    "openai_developer_kit",
    "rag",
]

__version__ = "0.1.2"


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        module = import_module(_MODULE_EXPORTS[name])
        globals()[name] = module
        return module

    if name in _ATTR_EXPORTS:
        module_name, attr_name = _ATTR_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
