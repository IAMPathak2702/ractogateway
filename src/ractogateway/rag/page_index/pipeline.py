"""PageIndexRAG — vectorless RAG using BM25 and a decision-tree index.

Unlike :class:`~ractogateway.rag.pipeline.RactoRAG`, this pipeline requires
**no embedding model** and **no vector store**.  It indexes documents at the
*page* level and retrieves using a two-stage approach:

1. **Decision index** (inverted keyword index) — narrows the full corpus to
   candidate pages that share at least one token with the query.
2. **BM25 scoring** — ranks the candidates with Okapi BM25 for accurate
   relevance ordering.

This makes it ideal for keyword-rich corpora (legal, technical, financial
documents) where exact term matching matters more than semantic similarity.

Quick start::

    from ractogateway.rag.page_index import PageIndexRAG

    rag = PageIndexRAG(llm_kit=my_kit)
    rag.ingest("report.pdf")
    response = rag.query("What were the Q3 revenue figures?")
    print(response.answer.content)

    # Retrieve without LLM
    results = rag.retrieve("revenue", top_k=5)
    for r in results:
        print(r.rank, r.score, r.entry.source, r.entry.page_number)
"""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any

from ractogateway._models.chat import ChatConfig
from ractogateway.prompts.engine import RactoPrompt
from ractogateway.rag._models.document import Document
from ractogateway.rag.page_index._bm25 import BM25Index, _DecisionIndex, extract_keywords
from ractogateway.rag.page_index._models import PageEntry, PageIndexResponse, PageIndexResult
from ractogateway.rag.processors.base import BaseProcessor
from ractogateway.rag.processors.cleaner import TextCleaner
from ractogateway.rag.readers.registry import FileReaderRegistry

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONTEXT_TEMPLATE = """\
Use the following retrieved page excerpts to answer the user's question.
If the excerpts do not contain enough information, say so clearly.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}"""

_DEFAULT_PAGE_RAG_PROMPT = RactoPrompt(
    role="You are a precise, factual question-answering assistant.",
    aim="Answer the user's question accurately based solely on the provided page excerpts.",
    constraints=[
        "Do not invent facts not present in the provided excerpts.",
        "If the context is insufficient, explicitly state that.",
        "Cite the source document and page number when possible.",
    ],
    tone="Clear, concise, and professional.",
    output_format="A direct answer, optionally with bullet points for multi-part questions.",
)

# Regex for detecting Markdown-style headings on a page
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Page splitter helpers
# ---------------------------------------------------------------------------

def _split_into_windows(
    text: str, page_size: int, page_overlap: int
) -> list[tuple[int | None, str]]:
    """Split *text* into fixed-size windows. Returns (page_number, content)."""
    windows: list[tuple[int | None, str]] = []
    start = 0
    idx = 1
    while start < len(text):
        end = min(start + page_size, len(text))
        windows.append((idx, text[start:end]))
        idx += 1
        start += page_size - page_overlap
        if end == len(text):
            break
    return windows


def _detect_section_title(text: str) -> str | None:
    """Return the first Markdown heading found in *text*, or ``None``."""
    m = _HEADING_RE.search(text)
    return m.group(1).strip() if m else None


def _require_pypdf() -> Any:
    try:
        import pypdf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for page-aware PDF ingestion in PageIndexRAG. "
            "Install it with:  pip install ractogateway[rag-pdf]"
        ) from exc
    return pypdf


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class PageIndexRAG:
    """Vectorless RAG pipeline that indexes documents at the page level.

    Parameters
    ----------
    llm_kit:
        Any RactoGateway developer kit (OpenAI, Anthropic, Google, Ollama,
        HuggingFace).  Required only for :meth:`query` / :meth:`aquery`.
        Pass ``None`` to use the pipeline in retrieve-only mode.
    processors:
        Text processors applied to each page before indexing.
        Defaults to ``[TextCleaner()]``.
    reader_registry:
        File reader registry used to load non-PDF documents.
        Defaults to a :class:`~ractogateway.rag.readers.registry.FileReaderRegistry`
        with all built-in readers registered.
    context_template:
        Jinja-style template with ``{context}`` and ``{question}`` placeholders
        used when building the LLM prompt.
    default_prompt:
        :class:`~ractogateway.prompts.engine.RactoPrompt` used for generation.
        Defaults to a built-in factual Q&A prompt.
    page_size:
        Maximum character length of each page window for non-PDF files
        (default 1 000).
    page_overlap:
        Character overlap between consecutive windows (default 100).
    k1:
        BM25 term-frequency saturation parameter (default 1.5).
    b:
        BM25 length-normalisation parameter (default 0.75).
    top_keywords:
        Number of top TF-weighted keywords to extract per page for the
        decision index (default 20).
    """

    def __init__(
        self,
        llm_kit: Any = None,
        *,
        processors: list[BaseProcessor] | None = None,
        reader_registry: FileReaderRegistry | None = None,
        context_template: str = _DEFAULT_CONTEXT_TEMPLATE,
        default_prompt: RactoPrompt | None = None,
        page_size: int = 1000,
        page_overlap: int = 100,
        k1: float = 1.5,
        b: float = 0.75,
        top_keywords: int = 20,
    ) -> None:
        self._llm_kit = llm_kit
        self._processors: list[BaseProcessor] = processors if processors is not None else [TextCleaner()]
        self._reader_registry = reader_registry or FileReaderRegistry()
        self._context_template = context_template
        self._default_prompt = default_prompt or _DEFAULT_PAGE_RAG_PROMPT
        self._page_size = page_size
        self._page_overlap = page_overlap
        self._top_keywords = top_keywords

        # In-process storage
        self._entries: dict[str, PageEntry] = {}       # entry_id → PageEntry
        self._bm25 = BM25Index(k1=k1, b=b)
        self._decision = _DecisionIndex()
        self._doc_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_processors(self, text: str) -> str:
        for proc in self._processors:
            text = proc.process(text)
        return text

    def _pages_from_pdf(self, path: str) -> list[tuple[int, str]]:
        """Extract text page-by-page from a PDF using pypdf."""
        pypdf = _require_pypdf()
        pages: list[tuple[int, str]] = []
        with pypdf.PdfReader(path) as reader:
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i, text))
        return pages

    def _pages_from_doc(self, doc: Document) -> list[tuple[int | None, str]]:
        """Produce (page_number, text) tuples from an arbitrary document."""
        return _split_into_windows(doc.content, self._page_size, self._page_overlap)

    def _build_entries(
        self,
        raw_pages: list[tuple[int | None, str]],
        source: str,
        doc_id: str,
        extra: dict[str, Any],
    ) -> list[PageEntry]:
        entries: list[PageEntry] = []
        for page_num, raw_text in raw_pages:
            processed = self._apply_processors(raw_text)
            if not processed:
                continue
            keywords = extract_keywords(processed, self._top_keywords)
            entry = PageEntry(
                page_number=page_num,
                content=processed,
                source=source,
                section_title=_detect_section_title(processed),
                keywords=keywords,
                doc_id=doc_id,
                char_count=len(processed),
                extra=extra,
            )
            entries.append(entry)
        return entries

    def _index_entries(self, entries: list[PageEntry]) -> None:
        for entry in entries:
            self._entries[entry.entry_id] = entry
            self._bm25.add(entry.entry_id, entry.content)
            self._decision.add(entry.entry_id, entry.keywords)

    def _build_context(self, results: list[PageIndexResult]) -> str:
        parts: list[str] = []
        for r in results:
            page_label = f"p.{r.entry.page_number}" if r.entry.page_number else f"window {r.rank}"
            title = f" — {r.entry.section_title}" if r.entry.section_title else ""
            parts.append(
                f"[{r.rank}] Source: {r.entry.source} ({page_label}){title}\n{r.entry.content}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Retrieval core (sync)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[PageIndexResult]:
        """Retrieve the most relevant pages for *query*.

        Uses two-stage retrieval: decision index (candidate selection) →
        BM25 scoring (ranking).

        Parameters
        ----------
        query:
            Natural-language question or keyword string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[PageIndexResult]
            Pages ranked by BM25 score (most relevant first).
        """
        from ractogateway.rag.page_index._bm25 import _tokenise  # noqa: PLC0415
        query_terms = _tokenise(query)
        candidates = self._decision.candidates(query_terms)
        if not candidates:
            # Full-scan fallback when no terms match the decision index
            candidates = None  # type: ignore[assignment]
        scored = self._bm25.score(query, candidates)
        results: list[PageIndexResult] = []
        for rank, (eid, score, matched) in enumerate(scored[:top_k], start=1):
            entry = self._entries[eid]
            results.append(
                PageIndexResult(entry=entry, score=score, rank=rank, matched_terms=matched)
            )
        return results

    async def aretrieve(self, query: str, top_k: int = 5) -> list[PageIndexResult]:
        """Async variant of :meth:`retrieve`."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.retrieve, query, top_k
        )

    # ------------------------------------------------------------------
    # Ingest — PDF
    # ------------------------------------------------------------------

    def _ingest_pdf(self, path: str, extra: dict[str, Any]) -> list[PageEntry]:
        doc_id = str(uuid.uuid4())
        self._doc_ids.add(doc_id)
        raw_pages = [(pn, txt) for pn, txt in self._pages_from_pdf(path)]
        entries = self._build_entries(raw_pages, path, doc_id, extra)
        self._index_entries(entries)
        return entries

    # ------------------------------------------------------------------
    # Ingest — generic (uses FileReaderRegistry)
    # ------------------------------------------------------------------

    def _ingest_generic(self, path: str, extra: dict[str, Any]) -> list[PageEntry]:
        doc = self._reader_registry.read(path)
        doc_id = doc.doc_id
        self._doc_ids.add(doc_id)
        raw_pages = self._pages_from_doc(doc)
        entries = self._build_entries(raw_pages, path, doc_id, extra)
        self._index_entries(entries)
        return entries

    # ------------------------------------------------------------------
    # Public ingest methods (sync)
    # ------------------------------------------------------------------

    def ingest(self, path: str, **metadata: Any) -> list[PageEntry]:
        """Read a file and add its pages to the index.

        PDFs are split page-by-page; all other file types are split into
        fixed-size character windows.

        Parameters
        ----------
        path:
            Absolute or relative path to the file.
        **metadata:
            Arbitrary key/value pairs stored in ``PageEntry.extra``.

        Returns
        -------
        list[PageEntry]
            All page entries created from this file.
        """
        p = Path(path)
        if p.suffix.lower() == ".pdf":
            return self._ingest_pdf(str(p.resolve()), metadata)
        return self._ingest_generic(str(p.resolve()), metadata)

    async def aingest(self, path: str, **metadata: Any) -> list[PageEntry]:
        """Async variant of :meth:`ingest`."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.ingest(path, **metadata)
        )

    def ingest_text(
        self, text: str, source: str = "manual", **metadata: Any
    ) -> list[PageEntry]:
        """Index raw text directly (no file I/O).

        Parameters
        ----------
        text:
            Plain text to index.
        source:
            Descriptive label stored in ``PageEntry.source``.
        **metadata:
            Arbitrary key/value pairs stored in ``PageEntry.extra``.
        """
        doc_id = str(uuid.uuid4())
        self._doc_ids.add(doc_id)
        doc = Document(content=text, source=source, doc_id=doc_id)
        raw_pages = self._pages_from_doc(doc)
        entries = self._build_entries(raw_pages, source, doc_id, metadata)
        self._index_entries(entries)
        return entries

    async def aingest_text(
        self, text: str, source: str = "manual", **metadata: Any
    ) -> list[PageEntry]:
        """Async variant of :meth:`ingest_text`."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.ingest_text(text, source, **metadata)
        )

    def ingest_dir(
        self, directory: str, pattern: str = "**/*", **metadata: Any
    ) -> list[PageEntry]:
        """Ingest all files matching *pattern* inside *directory*.

        Files that cannot be read are logged and skipped; the rest are
        indexed normally.

        Parameters
        ----------
        directory:
            Root directory to search.
        pattern:
            Glob pattern relative to *directory* (default ``"**/*"``).
        **metadata:
            Forwarded to every :meth:`ingest` call.
        """
        root = Path(directory)
        all_entries: list[PageEntry] = []
        for file_path in root.glob(pattern):
            if not file_path.is_file():
                continue
            try:
                all_entries.extend(self.ingest(str(file_path), **metadata))
            except Exception as exc:  # noqa: BLE001
                import logging  # noqa: PLC0415
                logging.getLogger(__name__).warning(
                    "PageIndexRAG: skipping %s — %s", file_path, exc
                )
        return all_entries

    async def aingest_dir(
        self, directory: str, pattern: str = "**/*", **metadata: Any
    ) -> list[PageEntry]:
        """Async variant of :meth:`ingest_dir`."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.ingest_dir(directory, pattern, **metadata)
        )

    # ------------------------------------------------------------------
    # RAG query (retrieve + generate)
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        prompt: RactoPrompt | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> PageIndexResponse:
        """Retrieve relevant pages and generate an answer with the LLM kit.

        Parameters
        ----------
        question:
            Natural-language question to answer.
        top_k:
            Number of pages to retrieve.
        prompt:
            Override the kit's default prompt for this call.
        temperature:
            Sampling temperature for generation.
        max_tokens:
            Maximum generation tokens.

        Returns
        -------
        PageIndexResponse
            Contains the generated answer, ranked sources, and the context
            string that was supplied to the model.

        Raises
        ------
        ValueError
            If no ``llm_kit`` was provided and generation is requested.
        """
        results = self.retrieve(question, top_k=top_k)
        context = self._build_context(results)
        context_used = self._context_template.format(context=context, question=question)

        answer = None
        if self._llm_kit is not None:
            active_prompt = prompt or self._default_prompt
            config = ChatConfig(
                user_message=context_used,
                prompt=active_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = self._llm_kit.chat(config)
        elif prompt is not None:
            raise ValueError(
                "A prompt was provided but no llm_kit is configured on this PageIndexRAG. "
                "Pass llm_kit=<kit> to the constructor."
            )

        return PageIndexResponse(
            answer=answer,
            sources=results,
            query=question,
            context_used=context_used,
        )

    async def aquery(
        self,
        question: str,
        *,
        top_k: int = 5,
        prompt: RactoPrompt | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> PageIndexResponse:
        """Async variant of :meth:`query`."""
        results = await self.aretrieve(question, top_k=top_k)
        context = self._build_context(results)
        context_used = self._context_template.format(context=context, question=question)

        answer = None
        if self._llm_kit is not None:
            active_prompt = prompt or self._default_prompt
            config = ChatConfig(
                user_message=context_used,
                prompt=active_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if hasattr(self._llm_kit, "achat"):
                answer = await self._llm_kit.achat(config)
            else:
                answer = await asyncio.get_event_loop().run_in_executor(
                    None, self._llm_kit.chat, config
                )
        elif prompt is not None:
            raise ValueError(
                "A prompt was provided but no llm_kit is configured on this PageIndexRAG. "
                "Pass llm_kit=<kit> to the constructor."
            )

        return PageIndexResponse(
            answer=answer,
            sources=results,
            query=question,
            context_used=context_used,
        )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all indexed entries and reset the pipeline to empty state."""
        self._entries.clear()
        self._bm25.clear()
        self._decision.clear()
        self._doc_ids.clear()

    @property
    def entry_count(self) -> int:
        """Total number of indexed page entries."""
        return len(self._entries)

    @property
    def document_count(self) -> int:
        """Number of distinct documents ingested."""
        return len(self._doc_ids)
