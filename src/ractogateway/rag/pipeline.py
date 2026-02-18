"""RactoRAG — the unified RAG pipeline.

Combines file reading, text processing, chunking, embedding, vector storage,
retrieval, and LLM-based generation into a single, ergonomic interface.

Example::

    from ractogateway import openai_developer_kit as opd
    from ractogateway.rag.pipeline import RactoRAG
    from ractogateway.rag.embedders.openai_embedder import OpenAIEmbedder
    from ractogateway.rag.stores.chroma_store import ChromaStore

    kit = opd.OpenAIDeveloperKit(model="gpt-4o")
    rag = RactoRAG(
        vector_store=ChromaStore(collection="docs"),
        embedder=OpenAIEmbedder(),
        llm_kit=kit,
    )
    rag.ingest("report.pdf")
    response = rag.query("What were the key findings?")
    print(response.answer.content)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ractogateway._models.chat import ChatConfig
from ractogateway.adapters.base import LLMResponse
from ractogateway.prompts.engine import RactoPrompt
from ractogateway.rag._models.document import Chunk, Document
from ractogateway.rag._models.retrieval import RAGResponse, RetrievalResult
from ractogateway.rag.chunkers.base import BaseChunker
from ractogateway.rag.chunkers.recursive_chunker import RecursiveChunker
from ractogateway.rag.embedders.base import BaseEmbedder
from ractogateway.rag.processors.base import BaseProcessor
from ractogateway.rag.processors.cleaner import TextCleaner
from ractogateway.rag.readers.registry import FileReaderRegistry
from ractogateway.rag.stores.base import BaseVectorStore

_DEFAULT_CONTEXT_TEMPLATE = """\
Use the following retrieved context to answer the user's question.
If the context does not contain enough information, say so clearly.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}"""

_DEFAULT_RAG_PROMPT = RactoPrompt(
    role="You are a precise, factual question-answering assistant.",
    aim="Answer the user's question accurately based solely on the provided context.",
    constraints=[
        "Do not invent facts not present in the context.",
        "If the context is insufficient, explicitly state that.",
        "Cite the source document when possible.",
    ],
    tone="Clear, concise, and professional.",
    output_format="A direct answer to the question, optionally with bullet points.",
)


class RactoRAG:
    """Production-grade RAG pipeline for RactoGateway.

    Parameters
    ----------
    vector_store:
        Any :class:`~ractogateway.rag.stores.base.BaseVectorStore` instance.
    embedder:
        Any :class:`~ractogateway.rag.embedders.base.BaseEmbedder` instance.
    chunker:
        How to split documents.  Defaults to
        :class:`~ractogateway.rag.chunkers.recursive_chunker.RecursiveChunker`
        with ``chunk_size=512, overlap=50``.
    processors:
        List of text processors applied to each chunk before embedding.
        Defaults to ``[TextCleaner()]``.
    llm_kit:
        Any developer kit (``OpenAIDeveloperKit``, ``GoogleDeveloperKit``, or
        ``AnthropicDeveloperKit``).  Required for :meth:`query` / :meth:`aquery`.
    context_template:
        Template string for injecting retrieved context into the LLM prompt.
        Must contain ``{context}`` and ``{question}`` placeholders.
    reader_registry:
        Custom :class:`~ractogateway.rag.readers.registry.FileReaderRegistry`.
        Defaults to a registry with all built-in readers.
    default_prompt:
        RACTO prompt used for generation.  Falls back to a built-in RAG prompt.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        *,
        chunker: BaseChunker | None = None,
        processors: list[BaseProcessor] | None = None,
        llm_kit: Any | None = None,
        context_template: str | None = None,
        reader_registry: FileReaderRegistry | None = None,
        default_prompt: RactoPrompt | None = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._chunker = chunker or RecursiveChunker(chunk_size=512, overlap=50)
        self._processors: list[BaseProcessor] = processors if processors is not None else [TextCleaner()]
        self._llm_kit = llm_kit
        self._context_template = context_template or _DEFAULT_CONTEXT_TEMPLATE
        self._reader_registry = reader_registry or FileReaderRegistry()
        self._default_prompt = default_prompt or _DEFAULT_RAG_PROMPT

    # ==================================================================
    # Ingest
    # ==================================================================

    def ingest(self, path: str | Path, **metadata: Any) -> list[Chunk]:
        """Read, chunk, embed, and store a single file.

        Parameters
        ----------
        path:
            Path to the file to ingest.
        **metadata:
            Extra metadata merged into each chunk's ``ChunkMetadata.extra``.

        Returns
        -------
        list[Chunk]
            The chunks that were added to the vector store.
        """
        doc = self._reader_registry.read(path)
        if metadata:
            doc.metadata.update(metadata)
        return self._process_document(doc)

    def ingest_dir(
        self,
        directory: str | Path,
        pattern: str = "**/*",
        **metadata: Any,
    ) -> list[Chunk]:
        """Recursively ingest all supported files in a directory.

        Parameters
        ----------
        directory:
            Root directory to scan.
        pattern:
            Glob pattern relative to *directory*.
        **metadata:
            Extra metadata merged into every chunk.

        Returns
        -------
        list[Chunk]
            All chunks added across all ingested files.
        """
        root = Path(directory)
        all_chunks: list[Chunk] = []
        for file_path in root.glob(pattern):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self._reader_registry.supported_extensions:
                continue
            try:
                chunks = self.ingest(file_path, **metadata)
                all_chunks.extend(chunks)
            except Exception as exc:  # noqa: BLE001
                # Log and continue rather than aborting the entire batch
                import warnings  # noqa: PLC0415

                warnings.warn(f"Failed to ingest {file_path}: {exc}", stacklevel=2)
        return all_chunks

    def ingest_text(
        self,
        text: str,
        source: str = "manual",
        **metadata: Any,
    ) -> list[Chunk]:
        """Ingest a raw text string directly (no file needed).

        Parameters
        ----------
        text:
            The text content to ingest.
        source:
            A label identifying the source (stored in metadata).
        **metadata:
            Extra metadata merged into each chunk.
        """
        doc = Document(content=text, source=source, metadata=metadata)
        return self._process_document(doc)

    async def aingest(self, path: str | Path, **metadata: Any) -> list[Chunk]:
        """Async variant of :meth:`ingest`."""
        doc = self._reader_registry.read(path)
        if metadata:
            doc.metadata.update(metadata)
        return await self._aprocess_document(doc)

    async def aingest_dir(
        self,
        directory: str | Path,
        pattern: str = "**/*",
        **metadata: Any,
    ) -> list[Chunk]:
        """Async variant of :meth:`ingest_dir`."""
        root = Path(directory)
        tasks = []
        for file_path in root.glob(pattern):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self._reader_registry.supported_extensions:
                continue
            tasks.append(self.aingest(file_path, **metadata))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_chunks: list[Chunk] = []
        for r in results:
            if isinstance(r, list):
                all_chunks.extend(r)
        return all_chunks

    async def aingest_text(
        self,
        text: str,
        source: str = "manual",
        **metadata: Any,
    ) -> list[Chunk]:
        """Async variant of :meth:`ingest_text`."""
        doc = Document(content=text, source=source, metadata=metadata)
        return await self._aprocess_document(doc)

    # ==================================================================
    # Retrieve
    # ==================================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Embed *query* and retrieve the top-k most relevant chunks.

        Parameters
        ----------
        query:
            Natural-language question or search phrase.
        top_k:
            Number of results to return.
        filters:
            Optional metadata filters (store-specific format).

        Returns
        -------
        list[RetrievalResult]
            Ranked results (rank 1 = most relevant).
        """
        embeddings = self._embedder.embed([query])
        return self._store.search(embeddings[0], top_k=top_k, filters=filters)

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Async variant of :meth:`retrieve`."""
        embeddings = await self._embedder.aembed([query])
        return self._store.search(embeddings[0], top_k=top_k, filters=filters)

    # ==================================================================
    # RAG Query (retrieve + generate)
    # ==================================================================

    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        prompt: RactoPrompt | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> RAGResponse:
        """Retrieve relevant chunks and generate an answer.

        Parameters
        ----------
        question:
            The user's question.
        top_k:
            Number of context chunks to retrieve.
        filters:
            Optional metadata filters for retrieval.
        prompt:
            Override the default RACTO prompt for generation.
        temperature:
            LLM temperature (default ``0.0`` for factual answers).
        max_tokens:
            Maximum tokens in the generated answer.

        Returns
        -------
        RAGResponse
            Contains the generated answer plus the retrieved source chunks.

        Raises
        ------
        RuntimeError
            If no ``llm_kit`` was provided.
        """
        self._require_llm_kit()
        sources = self.retrieve(question, top_k=top_k, filters=filters)
        context = self._build_context(sources)
        user_message = self._context_template.format(context=context, question=question)
        config = ChatConfig(
            user_message=user_message,
            prompt=prompt or self._default_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer: LLMResponse = self._llm_kit.chat(config)
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            context_used=context,
        )

    async def aquery(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        prompt: RactoPrompt | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> RAGResponse:
        """Async variant of :meth:`query`."""
        self._require_llm_kit()
        sources = await self.aretrieve(question, top_k=top_k, filters=filters)
        context = self._build_context(sources)
        user_message = self._context_template.format(context=context, question=question)
        config = ChatConfig(
            user_message=user_message,
            prompt=prompt or self._default_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer: LLMResponse = await self._llm_kit.achat(config)
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            context_used=context,
        )

    # ==================================================================
    # Convenience properties
    # ==================================================================

    @property
    def store(self) -> BaseVectorStore:
        """The underlying vector store."""
        return self._store

    @property
    def embedder(self) -> BaseEmbedder:
        """The underlying embedder."""
        return self._embedder

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        return self._store.count()

    def clear(self) -> None:
        """Remove all indexed chunks from the vector store."""
        self._store.clear()

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _process_document(self, doc: Document) -> list[Chunk]:
        """Chunk → process → embed → store."""
        chunks = self._chunker.chunk(doc)
        chunks = self._apply_processors(chunks)
        chunks = self._embed_chunks(chunks)
        self._store.add(chunks)
        return chunks

    async def _aprocess_document(self, doc: Document) -> list[Chunk]:
        """Async chunk → process → embed → store."""
        chunks = self._chunker.chunk(doc)
        chunks = self._apply_processors(chunks)
        chunks = await self._aembed_chunks(chunks)
        self._store.add(chunks)
        return chunks

    def _apply_processors(self, chunks: list[Chunk]) -> list[Chunk]:
        if not self._processors:
            return chunks
        processed: list[Chunk] = []
        for chunk in chunks:
            text = chunk.content
            for proc in self._processors:
                text = proc.process(text)
            processed.append(chunk.model_copy(update={"content": text}))
        return processed

    def _embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed(texts)
        return [
            chunk.model_copy(update={"embedding": emb})
            for chunk, emb in zip(chunks, embeddings)
        ]

    async def _aembed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        texts = [c.content for c in chunks]
        embeddings = await self._embedder.aembed(texts)
        return [
            chunk.model_copy(update={"embedding": emb})
            for chunk, emb in zip(chunks, embeddings)
        ]

    def _build_context(self, results: list[RetrievalResult]) -> str:
        parts: list[str] = []
        for i, result in enumerate(results, start=1):
            source = result.chunk.metadata.source
            page = result.chunk.metadata.page
            loc = f" (page {page})" if page else ""
            parts.append(
                f"[{i}] Source: {source}{loc}\n{result.chunk.content}"
            )
        return "\n\n".join(parts)

    def _require_llm_kit(self) -> None:
        if self._llm_kit is None:
            raise RuntimeError(
                "RactoRAG.query() requires an llm_kit. "
                "Pass one at construction time: RactoRAG(..., llm_kit=kit)"
            )
