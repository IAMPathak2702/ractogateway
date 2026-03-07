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

    from ractogateway import openai_developer_kit as gpt
    from ractogateway.rag.page_index import PageIndexRAG

    # 1. Setup
    kit = gpt.Chat(model="gpt-4o-mini")
    rag = PageIndexRAG(llm_kit=kit)

    # 2. Ingest
    rag.ingest("report.pdf")

    # 3. Query
    response = rag.query("What were the Q3 revenue figures?")
    print(response.answer.content)

    # Retrieve without LLM
    results = rag.retrieve("revenue", top_k=5)
    for r in results:
        print(r.rank, r.score, r.entry.source, r.entry.page_number)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ractogateway._models.chat import ChatConfig
from ractogateway.prompts.engine import RactoPrompt
from ractogateway.rag._models.document import Document
from ractogateway.rag.page_index._bm25 import (
    BM25Index,
    _DecisionIndex,
    extract_keywords,
)
from ractogateway.rag.page_index._models import (
    PageEntry,
    PageIndexResponse,
    PageIndexResult,
)
from ractogateway.rag.page_index._ocr import BaseOcrBackend
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
        import pypdf
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for page-aware PDF ingestion in PageIndexRAG. "
            "Install it with:  pip install ractogateway[rag-pdf]"
        ) from exc
    return pypdf


def _require_pdf2image() -> Any:
    try:
        import pdf2image
    except ImportError as exc:
        raise ImportError(
            "pdf2image is required for OCR-based PDF ingestion. "
            "Install it with:  pip install ractogateway[rag-ocr-pdf]"
        ) from exc
    return pdf2image


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        processors: Sequence[BaseProcessor] | None = None,
        reader_registry: FileReaderRegistry | None = None,
        context_template: str = _DEFAULT_CONTEXT_TEMPLATE,
        default_prompt: RactoPrompt | None = None,
        page_size: int = 1000,
        page_overlap: int = 100,
        k1: float = 1.5,
        b: float = 0.75,
        top_keywords: int = 20,
        ocr_backend: BaseOcrBackend | None = None,
        ocr_fallback: bool = True,
        min_ocr_confidence: float = 0.0,
    ) -> None:
        self._llm_kit = llm_kit
        self._processors: list[BaseProcessor] = list(
            processors if processors is not None else [TextCleaner()]
        )
        self._reader_registry = reader_registry or FileReaderRegistry()
        self._context_template = context_template
        self._default_prompt = default_prompt or _DEFAULT_PAGE_RAG_PROMPT
        self._page_size = page_size
        self._page_overlap = page_overlap
        self._top_keywords = top_keywords
        self._ocr_backend = ocr_backend
        self._ocr_fallback = ocr_fallback
        self._min_ocr_confidence = min_ocr_confidence

        # In-process storage
        self._entries: dict[str, PageEntry] = {}  # entry_id → PageEntry
        self._bm25 = BM25Index(k1=k1, b=b)
        self._decision = _DecisionIndex()
        self._doc_ids: set[str] = set()
        # Deduplication: SHA-256 of raw file bytes → doc_id
        self._file_hashes: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_processors(self, text: str) -> str:
        for proc in self._processors:
            text = proc.process(text)
        return text

    def _ocr_page_image(
        self, image_bytes: bytes, mime_type: str = "image/png"
    ) -> tuple[str, float | None]:
        """Run OCR on a single page image. Returns (text, confidence|None)."""
        assert self._ocr_backend is not None  # noqa: S101
        backend = self._ocr_backend
        # TesseractOcrBackend exposes confidence natively
        if hasattr(backend, "extract_with_confidence"):
            text, conf = backend.extract_with_confidence(image_bytes)
            return text, conf
        return backend.extract_text(image_bytes, mime_type), None

    def _pages_from_pdf(self, path: str) -> list[tuple[int, str, bool, float | None]]:
        """Extract text page-by-page from a PDF.

        Returns list of ``(page_number, text, ocr_applied, ocr_confidence)``.
        When a page has no embedded text and an OCR backend is configured,
        the page is rendered to an image and OCR'd automatically.
        """
        pypdf = _require_pypdf()
        pages: list[tuple[int, str, bool, float | None]] = []
        with pypdf.PdfReader(path) as reader:
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i, text, False, None))
                elif self._ocr_backend is not None and self._ocr_fallback:
                    # Render page to PNG and OCR it
                    pdf2image = _require_pdf2image()
                    images = pdf2image.convert_from_path(
                        path, first_page=i, last_page=i, fmt="png"
                    )
                    if images:
                        import io  # noqa: PLC0415

                        buf = io.BytesIO()
                        images[0].save(buf, format="PNG")
                        ocr_text, conf = self._ocr_page_image(buf.getvalue())
                        if ocr_text.strip():
                            if conf is None or conf >= self._min_ocr_confidence:
                                pages.append((i, ocr_text, True, conf))
        return pages

    def _pages_from_doc(self, doc: Document) -> list[tuple[int | None, str]]:
        """Produce (page_number, text) tuples from an arbitrary document."""
        return _split_into_windows(doc.content, self._page_size, self._page_overlap)

    def _build_entries(
        self,
        raw_pages: list[tuple[int | None, str]] | list[tuple[int, str, bool, float | None]],
        source: str,
        doc_id: str,
        extra: dict[str, Any],
    ) -> list[PageEntry]:
        entries: list[PageEntry] = []
        for item in raw_pages:
            if len(item) == 4:  # type: ignore[arg-type]
                page_num, raw_text, ocr_applied, ocr_conf = item  # type: ignore[misc]
            else:
                page_num, raw_text = item[0], item[1]  # type: ignore[misc]
                ocr_applied, ocr_conf = False, None
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
                ocr_applied=ocr_applied,
                ocr_confidence=ocr_conf,
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
            page_label = (
                f"p.{r.entry.page_number}"
                if r.entry.page_number
                else f"window {r.rank}"
            )
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
        from ractogateway.rag.page_index._bm25 import _tokenise

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
                PageIndexResult(
                    entry=entry, score=score, rank=rank, matched_terms=matched
                )
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

    def _file_hash(self, path: str) -> str:
        return _sha256(Path(path).read_bytes())

    def _ingest_pdf(self, path: str, extra: dict[str, Any]) -> list[PageEntry]:
        file_hash = self._file_hash(path)
        if file_hash in self._file_hashes:
            # Already indexed — return cached entries
            cached_doc_id = self._file_hashes[file_hash]
            return [e for e in self._entries.values() if e.doc_id == cached_doc_id]
        doc_id = str(uuid.uuid4())
        self._doc_ids.add(doc_id)
        self._file_hashes[file_hash] = doc_id
        raw_pages = self._pages_from_pdf(path)
        entries = self._build_entries(raw_pages, path, doc_id, extra)  # type: ignore[arg-type]
        self._index_entries(entries)
        return entries

    # ------------------------------------------------------------------
    # Ingest — generic (uses FileReaderRegistry)
    # ------------------------------------------------------------------

    def _ingest_generic(self, path: str, extra: dict[str, Any]) -> list[PageEntry]:
        file_hash = self._file_hash(path)
        if file_hash in self._file_hashes:
            cached_doc_id = self._file_hashes[file_hash]
            return [e for e in self._entries.values() if e.doc_id == cached_doc_id]
        doc = self._reader_registry.read(path)
        doc_id = doc.doc_id
        self._doc_ids.add(doc_id)
        self._file_hashes[file_hash] = doc_id
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
        self,
        directory: str,
        pattern: str = "**/*",
        *,
        on_progress: Callable[[int, int], None] | None = None,
        **metadata: Any,
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
        on_progress:
            Optional callback ``(done, total) -> None`` called after each file
            is processed (or skipped).  Useful for progress bars.
        **metadata:
            Forwarded to every :meth:`ingest` call.
        """
        import logging  # noqa: PLC0415

        root = Path(directory)
        files = [p for p in root.glob(pattern) if p.is_file()]
        total = len(files)
        all_entries: list[PageEntry] = []
        for done, file_path in enumerate(files, start=1):
            try:
                all_entries.extend(self.ingest(str(file_path), **metadata))
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "PageIndexRAG: skipping %s - %s", file_path, exc
                )
            if on_progress is not None:
                on_progress(done, total)
        return all_entries

    async def aingest_dir(
        self,
        directory: str,
        pattern: str = "**/*",
        *,
        max_concurrent: int = 4,
        on_progress: Callable[[int, int], None] | None = None,
        **metadata: Any,
    ) -> list[PageEntry]:
        """Async parallel variant of :meth:`ingest_dir`.

        Parameters
        ----------
        directory:
            Root directory to search.
        pattern:
            Glob pattern relative to *directory* (default ``"**/*"``).
        max_concurrent:
            Maximum number of files ingested concurrently (default 4).
        on_progress:
            Optional callback ``(done, total) -> None`` called after each file
            finishes (thread-safe; called from the event loop).
        **metadata:
            Forwarded to every :meth:`aingest` call.
        """
        import logging  # noqa: PLC0415

        root = Path(directory)
        files = [p for p in root.glob(pattern) if p.is_file()]
        total = len(files)
        all_entries: list[PageEntry] = []
        sem = asyncio.Semaphore(max_concurrent)
        done_count = 0

        async def _ingest_one(fp: Path) -> list[PageEntry]:
            nonlocal done_count
            async with sem:
                try:
                    result = await self.aingest(str(fp), **metadata)
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "PageIndexRAG: skipping %s - %s", fp, exc
                    )
                    result = []
                done_count += 1
                if on_progress is not None:
                    on_progress(done_count, total)
                return result

        results = await asyncio.gather(*[_ingest_one(fp) for fp in files])
        for batch in results:
            all_entries.extend(batch)
        return all_entries

    # ------------------------------------------------------------------
    # Aliases & Compatibility
    # ------------------------------------------------------------------

    def add_document(self, path: str, **metadata: Any) -> list[PageEntry]:
        """Alias for :meth:`ingest`."""
        return self.ingest(path, **metadata)

    def add_texts(
        self, texts: Sequence[str], source: str = "manual", **metadata: Any
    ) -> list[PageEntry]:
        """Ingest a list of text strings."""
        entries: list[PageEntry] = []
        for t in texts:
            entries.extend(self.ingest_text(t, source=source, **metadata))
        return entries

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        prompt: RactoPrompt | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> PageIndexResponse:
        """Alias for :meth:`query`."""
        return self.query(
            query,
            top_k=top_k,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
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

    def remove_document(self, doc_id: str) -> int:
        """Remove all pages belonging to *doc_id* from the index.

        Parameters
        ----------
        doc_id:
            The ``doc_id`` value from any :class:`PageEntry` returned during
            ingestion.

        Returns
        -------
        int
            Number of page entries removed.
        """
        to_remove = [eid for eid, e in self._entries.items() if e.doc_id == doc_id]
        for eid in to_remove:
            self._decision.remove(eid)
            self._bm25.remove(eid)
            del self._entries[eid]
        self._doc_ids.discard(doc_id)
        # Remove the file hash mapping for this doc
        self._file_hashes = {h: d for h, d in self._file_hashes.items() if d != doc_id}
        return len(to_remove)

    def clear(self) -> None:
        """Remove all indexed entries and reset the pipeline to empty state."""
        self._entries.clear()
        self._bm25.clear()
        self._decision.clear()
        self._doc_ids.clear()
        self._file_hashes.clear()

    # ------------------------------------------------------------------
    # Persistence — save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the full index to a JSON file.

        The saved file contains all :class:`PageEntry` records, BM25 term
        weights, and deduplication hashes.  Reload with :meth:`load`.

        Parameters
        ----------
        path:
            Destination file path (will be created or overwritten).
        """
        import collections  # noqa: PLC0415

        data: dict[str, Any] = {
            "version": 1,
            "entries": [e.model_dump() for e in self._entries.values()],
            "bm25": {
                "k1": self._bm25._k1,
                "b": self._bm25._b,
                "corpus": {
                    eid: dict(counter)
                    for eid, counter in self._bm25._corpus.items()
                },
                "lengths": self._bm25._lengths,
                "df": dict(self._bm25._df),
                "avg_dl": self._bm25._avg_dl,
            },
            "file_hashes": self._file_hashes,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "PageIndexRAG":
        """Load a previously saved index from *path*.

        Parameters
        ----------
        path:
            JSON file written by :meth:`save`.
        **kwargs:
            Forwarded to the constructor (e.g. ``llm_kit=kit``).

        Returns
        -------
        PageIndexRAG
            A new instance with the index fully restored.
        """
        import collections  # noqa: PLC0415

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        instance = cls(**kwargs)

        # Restore entries
        for entry_data in raw["entries"]:
            entry = PageEntry(**entry_data)
            instance._entries[entry.entry_id] = entry
            instance._doc_ids.add(entry.doc_id)
            instance._decision.add(entry.entry_id, entry.keywords)

        # Restore BM25 state
        bm25_data = raw["bm25"]
        instance._bm25._k1 = bm25_data["k1"]
        instance._bm25._b = bm25_data["b"]
        instance._bm25._corpus = {
            eid: collections.Counter(tf) for eid, tf in bm25_data["corpus"].items()
        }
        instance._bm25._lengths = bm25_data["lengths"]
        instance._bm25._df = collections.Counter(bm25_data["df"])
        instance._bm25._avg_dl = bm25_data["avg_dl"]

        # Restore dedup hashes
        instance._file_hashes = raw.get("file_hashes", {})
        return instance

    @property
    def entry_count(self) -> int:
        """Total number of indexed page entries."""
        return len(self._entries)

    @property
    def document_count(self) -> int:
        """Number of distinct documents ingested."""
        return len(self._doc_ids)
