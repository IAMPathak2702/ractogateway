"""RactoGateway — Unified AI SDK with anti-hallucination prompting via RACTO.

Developer Kits (primary API)::

    from ractogateway import openai_developer_kit as opd
    from ractogateway import google_developer_kit as god
    from ractogateway import anthropic_developer_kit as anth

Core utilities::

    from ractogateway import RactoPrompt, ToolRegistry, tool, Gateway

RAG (Retrieval-Augmented Generation)::

    from ractogateway import rag
    from ractogateway.rag.pipeline import RactoRAG
    from ractogateway.rag.embedders.openai_embedder import OpenAIEmbedder
    from ractogateway.rag.stores.chroma_store import ChromaStore

    pipeline = RactoRAG(
        vector_store=ChromaStore(collection="docs"),
        embedder=OpenAIEmbedder(),
        llm_kit=kit,
    )
    pipeline.ingest("report.pdf")
    response = pipeline.query("What are the key findings?")
"""

# Developer Kit subpackages — available as module-level imports.
# Each kit lazily imports its provider SDK only when instantiated.
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

__all__ = [
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
    # RAG — Pipeline
    "RactoRAG",
    # RAG — Chunkers
    "FixedChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    # RAG — Embedders
    "GoogleEmbedder",
    "OpenAIEmbedder",
    "VoyageEmbedder",
    # RAG — Processors
    "Lemmatizer",
    "ProcessingPipeline",
    "TextCleaner",
    # RAG — Readers
    "FileReaderRegistry",
    # RAG — Stores
    "ChromaStore",
    "FAISSStore",
    "InMemoryVectorStore",
    "MilvusStore",
    "PGVectorStore",
    "PineconeStore",
    "QdrantStore",
    "WeaviateStore",
    # Sub-packages
    "anthropic_developer_kit",
    "finetune",
    "google_developer_kit",
    "openai_developer_kit",
    "rag",
]
__version__ = "0.1.1"
