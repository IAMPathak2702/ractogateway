"""RactoGateway - unified AI SDK with anti-hallucination prompting via RACTO.

The package surface stays import-friendly while deferring heavy imports until a
specific symbol is accessed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ractogateway._version import __version__

if TYPE_CHECKING:
    from ractogateway import (
        anthropic_developer_kit,
        batch,
        cache,
        celery,
        finetune,
        google_developer_kit,
        huggingface_developer_kit,
        kafka,
        mail,
        mcp,
        ollama_developer_kit,
        openai_developer_kit,
        pipelines,
        rag,
        redis,
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
    from ractogateway.celery._models import RetryConfig, TaskResult, TaskStatus
    from ractogateway.celery.worker import RactoCeleryWorker
    from ractogateway.exceptions import (
        RactoGatewayAPIError,
        RactoGatewayAuthError,
        RactoGatewayError,
        RactoGatewayTimeoutError,
        ResponseModelValidationError,
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
    from ractogateway.kafka._models import (
        KafkaAuditEvent,
        KafkaConsumerConfig,
        KafkaMessage,
        KafkaProducerConfig,
        KafkaProduceResult,
        KafkaPublishRequest,
        KafkaStreamConfig,
    )
    from ractogateway.kafka.audit import KafkaAuditLogger
    from ractogateway.kafka.consumer import KafkaConsumerClient
    from ractogateway.kafka.producer import KafkaProducerClient
    from ractogateway.kafka.stream import KafkaStreamProcessor
    from ractogateway.mcp._models import MCPClientConfig, MCPServerConfig, MCPToolResult
    from ractogateway.mcp.agent import MCPAgent
    from ractogateway.mcp.client import RactoMCPClient
    from ractogateway.mcp.multi_client import MCPMultiClient
    from ractogateway.mcp.server import RactoMCPServer
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
    from ractogateway.redis._models import ChatMemoryConfig, RateLimitConfig
    from ractogateway.redis.chat_memory import RedisChatMemory
    from ractogateway.redis.exact_cache import RedisExactCache
    from ractogateway.redis.rate_limiter import RedisRateLimiter
    from ractogateway.routing._models import RoutingTier
    from ractogateway.routing.router import CostAwareRouter
    from ractogateway.pipelines.sql_analyst._models import (
        PipelineUsage,
        RateLimitExceededError,
        ReadOnlyViolationError,
        SQLAnalystResult,
    )
    from ractogateway.pipelines.sql_analyst._viz import ChartSpec
    from ractogateway.pipelines.sql_analyst.pipeline import (
        AsyncSQLAnalystPipeline,
        SQLAnalystPipeline,
    )
    from ractogateway.pipelines.list_classifier._models import (
        AuditEntry,
        ClassifierRateLimitExceededError,
        ClassifierResult,
        ClassifierUsage,
    )
    from ractogateway.pipelines.list_classifier.pipeline import (
        AsyncListClassifierPipeline,
        ListClassifierPipeline,
    )
    from ractogateway.tools.registry import ToolRegistry, tool
    from ractogateway.truncation._models import TruncationConfig
    from ractogateway.truncation.truncator import TokenTruncator

_MODULE_EXPORTS: dict[str, str] = {
    "anthropic_developer_kit": "ractogateway.anthropic_developer_kit",
    "batch": "ractogateway.batch",
    "cache": "ractogateway.cache",
    "celery": "ractogateway.celery",
    "finetune": "ractogateway.finetune",
    "google_developer_kit": "ractogateway.google_developer_kit",
    "huggingface_developer_kit": "ractogateway.huggingface_developer_kit",
    "kafka": "ractogateway.kafka",
    "local": "ractogateway.local",
    "mail": "ractogateway.mail",
    "mcp": "ractogateway.mcp",
    "ollama_developer_kit": "ractogateway.ollama_developer_kit",
    "openai_developer_kit": "ractogateway.openai_developer_kit",
    "pipelines": "ractogateway.pipelines",
    "rag": "ractogateway.rag",
    "redis": "ractogateway.redis",
    "routing": "ractogateway.routing",
    "truncation": "ractogateway.truncation",
}

_ATTR_EXPORTS: dict[str, tuple[str, str]] = {
    # Exceptions
    "RactoGatewayError": ("ractogateway.exceptions", "RactoGatewayError"),
    "RactoGatewayTimeoutError": ("ractogateway.exceptions", "RactoGatewayTimeoutError"),
    "RactoGatewayAPIError": ("ractogateway.exceptions", "RactoGatewayAPIError"),
    "RactoGatewayAuthError": ("ractogateway.exceptions", "RactoGatewayAuthError"),
    "ResponseModelValidationError": (
        "ractogateway.exceptions",
        "ResponseModelValidationError",
    ),
    # Fine-tuning
    "AnthropicFineTuner": ("ractogateway.finetune", "AnthropicFineTuner"),
    "GeminiFineTuner": ("ractogateway.finetune", "GeminiFineTuner"),
    "OpenAIFineTuner": ("ractogateway.finetune", "OpenAIFineTuner"),
    "RactoDataset": ("ractogateway.finetune", "RactoDataset"),
    "RactoTrainingExample": ("ractogateway.finetune", "RactoTrainingExample"),
    "RactoTrainingMessage": ("ractogateway.finetune", "RactoTrainingMessage"),
    # MCP
    "MCPAgent": ("ractogateway.mcp.agent", "MCPAgent"),
    "MCPClientConfig": ("ractogateway.mcp._models", "MCPClientConfig"),
    "MCPMultiClient": ("ractogateway.mcp.multi_client", "MCPMultiClient"),
    "MCPServerConfig": ("ractogateway.mcp._models", "MCPServerConfig"),
    "MCPToolResult": ("ractogateway.mcp._models", "MCPToolResult"),
    "RactoMCPClient": ("ractogateway.mcp.client", "RactoMCPClient"),
    "RactoMCPServer": ("ractogateway.mcp.server", "RactoMCPServer"),
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
    # Redis infrastructure
    "ChatMemoryConfig": ("ractogateway.redis._models", "ChatMemoryConfig"),
    "RateLimitConfig": ("ractogateway.redis._models", "RateLimitConfig"),
    "RedisChatMemory": ("ractogateway.redis.chat_memory", "RedisChatMemory"),
    "RedisExactCache": ("ractogateway.redis.exact_cache", "RedisExactCache"),
    "RedisRateLimiter": ("ractogateway.redis.rate_limiter", "RedisRateLimiter"),
    # Kafka streaming
    "KafkaAuditEvent": ("ractogateway.kafka._models", "KafkaAuditEvent"),
    "KafkaAuditLogger": ("ractogateway.kafka.audit", "KafkaAuditLogger"),
    "KafkaConsumerClient": ("ractogateway.kafka.consumer", "KafkaConsumerClient"),
    "KafkaConsumerConfig": ("ractogateway.kafka._models", "KafkaConsumerConfig"),
    "KafkaMessage": ("ractogateway.kafka._models", "KafkaMessage"),
    "KafkaProduceResult": ("ractogateway.kafka._models", "KafkaProduceResult"),
    "KafkaProducerClient": ("ractogateway.kafka.producer", "KafkaProducerClient"),
    "KafkaProducerConfig": ("ractogateway.kafka._models", "KafkaProducerConfig"),
    "KafkaPublishRequest": ("ractogateway.kafka._models", "KafkaPublishRequest"),
    "KafkaStreamConfig": ("ractogateway.kafka._models", "KafkaStreamConfig"),
    "KafkaStreamProcessor": ("ractogateway.kafka.stream", "KafkaStreamProcessor"),
    # Routing
    "CostAwareRouter": ("ractogateway.routing.router", "CostAwareRouter"),
    "RoutingTier": ("ractogateway.routing._models", "RoutingTier"),
    # Truncation
    "TokenTruncator": ("ractogateway.truncation.truncator", "TokenTruncator"),
    "TruncationConfig": ("ractogateway.truncation._models", "TruncationConfig"),
    # Celery task queue
    "RactoCeleryWorker": ("ractogateway.celery.worker", "RactoCeleryWorker"),
    "RetryConfig": ("ractogateway.celery._models", "RetryConfig"),
    "TaskResult": ("ractogateway.celery._models", "TaskResult"),
    "TaskStatus": ("ractogateway.celery._models", "TaskStatus"),
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
    # Pipelines — SQL Analyst
    "AsyncSQLAnalystPipeline": (
        "ractogateway.pipelines.sql_analyst.pipeline",
        "AsyncSQLAnalystPipeline",
    ),
    "ChartSpec": (
        "ractogateway.pipelines.sql_analyst._viz",
        "ChartSpec",
    ),
    "PipelineUsage": (
        "ractogateway.pipelines.sql_analyst._models",
        "PipelineUsage",
    ),
    "RateLimitExceededError": (
        "ractogateway.pipelines.sql_analyst._models",
        "RateLimitExceededError",
    ),
    "ReadOnlyViolationError": (
        "ractogateway.pipelines.sql_analyst._models",
        "ReadOnlyViolationError",
    ),
    "SQLAnalystPipeline": (
        "ractogateway.pipelines.sql_analyst.pipeline",
        "SQLAnalystPipeline",
    ),
    "SQLAnalystResult": (
        "ractogateway.pipelines.sql_analyst._models",
        "SQLAnalystResult",
    ),
    # Pipelines — List Classifier
    "AsyncListClassifierPipeline": (
        "ractogateway.pipelines.list_classifier.pipeline",
        "AsyncListClassifierPipeline",
    ),
    "AuditEntry": (
        "ractogateway.pipelines.list_classifier._models",
        "AuditEntry",
    ),
    "ClassifierRateLimitExceededError": (
        "ractogateway.pipelines.list_classifier._models",
        "ClassifierRateLimitExceededError",
    ),
    "ClassifierResult": (
        "ractogateway.pipelines.list_classifier._models",
        "ClassifierResult",
    ),
    "ClassifierUsage": (
        "ractogateway.pipelines.list_classifier._models",
        "ClassifierUsage",
    ),
    "ListClassifierPipeline": (
        "ractogateway.pipelines.list_classifier.pipeline",
        "ListClassifierPipeline",
    ),
}

__all__ = [
    # MCP — Model Context Protocol
    "MCPAgent",
    "MCPClientConfig",
    "MCPMultiClient",
    "MCPServerConfig",
    "MCPToolResult",
    "RactoMCPClient",
    "RactoMCPServer",
    # Exceptions
    "RactoGatewayError",
    "RactoGatewayTimeoutError",
    "RactoGatewayAPIError",
    "RactoGatewayAuthError",
    "ResponseModelValidationError",
    "__version__",
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
    # Redis infrastructure
    "ChatMemoryConfig",
    "RateLimitConfig",
    "RedisChatMemory",
    "RedisExactCache",
    "RedisRateLimiter",
    # Kafka streaming
    "KafkaAuditEvent",
    "KafkaAuditLogger",
    "KafkaConsumerClient",
    "KafkaConsumerConfig",
    "KafkaMessage",
    "KafkaProduceResult",
    "KafkaProducerClient",
    "KafkaProducerConfig",
    "KafkaPublishRequest",
    "KafkaStreamConfig",
    "KafkaStreamProcessor",
    # Routing
    "CostAwareRouter",
    "RoutingTier",
    # Truncation
    "TokenTruncator",
    "TruncationConfig",
    # Celery task queue
    "RactoCeleryWorker",
    "RetryConfig",
    "TaskResult",
    "TaskStatus",
    # Batch
    "AnthropicBatchProcessor",
    "BatchItem",
    "BatchJobInfo",
    "BatchResult",
    "BatchStatus",
    "OpenAIBatchProcessor",
    # Pipelines — SQL Analyst
    "AsyncSQLAnalystPipeline",
    "ChartSpec",
    "PipelineUsage",
    "RateLimitExceededError",
    "ReadOnlyViolationError",
    "SQLAnalystPipeline",
    "SQLAnalystResult",
    # Pipelines — List Classifier
    "AsyncListClassifierPipeline",
    "AuditEntry",
    "ClassifierRateLimitExceededError",
    "ClassifierResult",
    "ClassifierUsage",
    "ListClassifierPipeline",
    # Submodules
    "anthropic_developer_kit",
    "batch",
    "cache",
    "celery",
    "finetune",
    "google_developer_kit",
    "huggingface_developer_kit",
    "kafka",
    "local",
    "mail",
    "mcp",
    "ollama_developer_kit",
    "openai_developer_kit",
    "pipelines",
    "rag",
    "redis",
    "routing",
    "truncation",
]

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
