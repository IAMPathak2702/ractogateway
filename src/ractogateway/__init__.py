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
        batch,
        cache,
        finetune,
        google_developer_kit,
        openai_developer_kit,
        rag,
        routing,
        truncation,
    )
    from ractogateway.adapters.base import LLMResponse
    from ractogateway.batch._models import BatchItem, BatchJobInfo, BatchResult, BatchStatus
    from ractogateway.batch.anthropic_batch import AnthropicBatchProcessor
    from ractogateway.batch.openai_batch import OpenAIBatchProcessor
    from ractogateway.cache._models import CacheConfig, CacheStats
    from ractogateway.cache.exact_cache import ExactMatchCache
    from ractogateway.cache.semantic_cache import SemanticCache
    from ractogateway.exceptions import (
        RactoGatewayAPIError,
        RactoGatewayAuthError,
        RactoGatewayError,
        RactoGatewayTimeoutError,
    )
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
    from ractogateway.routing._models import RoutingTier
    from ractogateway.routing.router import CostAwareRouter
    from ractogateway.tools.registry import ToolRegistry, tool
    from ractogateway.truncation._models import TruncationConfig
    from ractogateway.truncation.truncator import TokenTruncator

_MODULE_EXPORTS: dict[str, str] = {
    "anthropic_developer_kit": "ractogateway.anthropic_developer_kit",
    "batch": "ractogateway.batch",
    "cache": "ractogateway.cache",
    "finetune": "ractogateway.finetune",
    "google_developer_kit": "ractogateway.google_developer_kit",
    "openai_developer_kit": "ractogateway.openai_developer_kit",
    "rag": "ractogateway.rag",
    "routing": "ractogateway.routing",
    "truncation": "ractogateway.truncation",
}

_ATTR_EXPORTS: dict[str, tuple[str, str]] = {
    # Exceptions
    "RactoGatewayError": ("ractogateway.exceptions", "RactoGatewayError"),
    "RactoGatewayTimeoutError": ("ractogateway.exceptions", "RactoGatewayTimeoutError"),
    "RactoGatewayAPIError": ("ractogateway.exceptions", "RactoGatewayAPIError"),
    "RactoGatewayAuthError": ("ractogateway.exceptions", "RactoGatewayAuthError"),
    # Fine-tuning
    "AnthropicFineTuner": ("ractogateway.finetune", "AnthropicFineTuner"),
    "GeminiFineTuner": ("ractogateway.finetune", "GeminiFineTuner"),
    "OpenAIFineTuner": ("ractogateway.finetune", "OpenAIFineTuner"),
    "RactoDataset": ("ractogateway.finetune", "RactoDataset"),
    "RactoTrainingExample": ("ractogateway.finetune", "RactoTrainingExample"),
    "RactoTrainingMessage": ("ractogateway.finetune", "RactoTrainingMessage"),
    # Core
    "Gateway": ("ractogateway.gateway.runner", "Gateway"),
    "LLMResponse": ("ractogateway.adapters.base", "LLMResponse"),
    "RactoFile": ("ractogateway.prompts.engine", "RactoFile"),
    "RactoPrompt": ("ractogateway.prompts.engine", "RactoPrompt"),
    "ToolRegistry": ("ractogateway.tools.registry", "ToolRegistry"),
    "tool": ("ractogateway.tools.registry", "tool"),
    # RAG pipeline
    "RactoRAG": ("ractogateway.rag.pipeline", "RactoRAG"),
    # RAG chunkers
    "FixedChunker": ("ractogateway.rag.chunkers.fixed_chunker", "FixedChunker"),
    "RecursiveChunker": ("ractogateway.rag.chunkers.recursive_chunker", "RecursiveChunker"),
    "SemanticChunker": ("ractogateway.rag.chunkers.semantic_chunker", "SemanticChunker"),
    "SentenceChunker": ("ractogateway.rag.chunkers.sentence_chunker", "SentenceChunker"),
    # RAG embedders
    "GoogleEmbedder": ("ractogateway.rag.embedders.google_embedder", "GoogleEmbedder"),
    "OpenAIEmbedder": ("ractogateway.rag.embedders.openai_embedder", "OpenAIEmbedder"),
    "VoyageEmbedder": ("ractogateway.rag.embedders.voyage_embedder", "VoyageEmbedder"),
    # RAG processors
    "Lemmatizer": ("ractogateway.rag.processors.lemmatizer", "Lemmatizer"),
    "ProcessingPipeline": ("ractogateway.rag.processors.pipeline", "ProcessingPipeline"),
    "TextCleaner": ("ractogateway.rag.processors.cleaner", "TextCleaner"),
    # RAG readers
    "FileReaderRegistry": ("ractogateway.rag.readers.registry", "FileReaderRegistry"),
    # RAG stores
    "ChromaStore": ("ractogateway.rag.stores.chroma_store", "ChromaStore"),
    "FAISSStore": ("ractogateway.rag.stores.faiss_store", "FAISSStore"),
    "InMemoryVectorStore": ("ractogateway.rag.stores.in_memory_store", "InMemoryVectorStore"),
    "MilvusStore": ("ractogateway.rag.stores.milvus_store", "MilvusStore"),
    "PGVectorStore": ("ractogateway.rag.stores.pgvector_store", "PGVectorStore"),
    "PineconeStore": ("ractogateway.rag.stores.pinecone_store", "PineconeStore"),
    "QdrantStore": ("ractogateway.rag.stores.qdrant_store", "QdrantStore"),
    "WeaviateStore": ("ractogateway.rag.stores.weaviate_store", "WeaviateStore"),
    # Cache
    "CacheConfig": ("ractogateway.cache._models", "CacheConfig"),
    "CacheStats": ("ractogateway.cache._models", "CacheStats"),
    "ExactMatchCache": ("ractogateway.cache.exact_cache", "ExactMatchCache"),
    "SemanticCache": ("ractogateway.cache.semantic_cache", "SemanticCache"),
    # Routing
    "CostAwareRouter": ("ractogateway.routing.router", "CostAwareRouter"),
    "RoutingTier": ("ractogateway.routing._models", "RoutingTier"),
    # Truncation
    "TokenTruncator": ("ractogateway.truncation.truncator", "TokenTruncator"),
    "TruncationConfig": ("ractogateway.truncation._models", "TruncationConfig"),
    # Batch
    "AnthropicBatchProcessor": (
        "ractogateway.batch.anthropic_batch",
        "AnthropicBatchProcessor",
    ),
    "BatchItem": ("ractogateway.batch._models", "BatchItem"),
    "BatchJobInfo": ("ractogateway.batch._models", "BatchJobInfo"),
    "BatchResult": ("ractogateway.batch._models", "BatchResult"),
    "BatchStatus": ("ractogateway.batch._models", "BatchStatus"),
    "OpenAIBatchProcessor": ("ractogateway.batch.openai_batch", "OpenAIBatchProcessor"),
}

__all__ = [
    # Exceptions
    "RactoGatewayError",
    "RactoGatewayTimeoutError",
    "RactoGatewayAPIError",
    "RactoGatewayAuthError",
    # Fine-tuning
    "AnthropicFineTuner",
    "GeminiFineTuner",
    "OpenAIFineTuner",
    "RactoDataset",
    "RactoTrainingExample",
    "RactoTrainingMessage",
    # Core
    "Gateway",
    "LLMResponse",
    "RactoFile",
    "RactoPrompt",
    "ToolRegistry",
    "tool",
    # RAG
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
    # Cache
    "CacheConfig",
    "CacheStats",
    "ExactMatchCache",
    "SemanticCache",
    # Routing
    "CostAwareRouter",
    "RoutingTier",
    # Truncation
    "TokenTruncator",
    "TruncationConfig",
    # Batch
    "AnthropicBatchProcessor",
    "BatchItem",
    "BatchJobInfo",
    "BatchResult",
    "BatchStatus",
    "OpenAIBatchProcessor",
    # Submodules
    "anthropic_developer_kit",
    "batch",
    "cache",
    "finetune",
    "google_developer_kit",
    "openai_developer_kit",
    "rag",
    "routing",
    "truncation",
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
