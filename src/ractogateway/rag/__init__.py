"""RactoGateway RAG — production-grade Retrieval-Augmented Generation.

Quick start::

    from ractogateway.rag.pipeline import RactoRAG
    from ractogateway.rag.embedders.openai_embedder import OpenAIEmbedder
    from ractogateway.rag.stores.chroma_store import ChromaStore

    rag = RactoRAG(
        vector_store=ChromaStore(collection="my_docs"),
        embedder=OpenAIEmbedder(),
        llm_kit=my_kit,
    )
    rag.ingest("docs/report.pdf")
    response = rag.query("What was the Q3 revenue?")

Components
----------
- **Readers**:   :mod:`ractogateway.rag.readers`
- **Chunkers**:  :mod:`ractogateway.rag.chunkers`
- **Processors**: :mod:`ractogateway.rag.processors`
- **Embedders**: :mod:`ractogateway.rag.embedders`
- **Stores**:    :mod:`ractogateway.rag.stores`
- **Pipeline**:  :class:`~ractogateway.rag.pipeline.RactoRAG`
"""

from ractogateway.rag import chunkers, embedders, processors, readers, stores
from ractogateway.rag._models.document import Chunk, ChunkMetadata, Document
from ractogateway.rag._models.retrieval import RAGResponse, RetrievalConfig, RetrievalResult
from ractogateway.rag.page_index import PageIndexRAG
from ractogateway.rag.page_index._models import PageEntry, PageIndexResponse, PageIndexResult
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

__all__ = [
    # Pipelines
    "RactoRAG",
    "PageIndexRAG",
    # PageIndex models
    "PageEntry",
    "PageIndexResult",
    "PageIndexResponse",
    # Models
    "Chunk",
    "ChunkMetadata",
    "Document",
    "RAGResponse",
    "RetrievalConfig",
    "RetrievalResult",
    # Chunkers
    "FixedChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    # Embedders
    "GoogleEmbedder",
    "OpenAIEmbedder",
    "VoyageEmbedder",
    # Processors
    "Lemmatizer",
    "ProcessingPipeline",
    "TextCleaner",
    # Readers
    "FileReaderRegistry",
    # Stores
    "ChromaStore",
    "FAISSStore",
    "InMemoryVectorStore",
    "MilvusStore",
    "PGVectorStore",
    "PineconeStore",
    "QdrantStore",
    "WeaviateStore",
    # Sub-packages
    "chunkers",
    "embedders",
    "processors",
    "readers",
    "stores",
]
