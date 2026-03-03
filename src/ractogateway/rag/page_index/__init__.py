"""PageIndexRAG — vectorless, page-level BM25 retrieval.

Quick start::

    from ractogateway.rag.page_index import PageIndexRAG

    rag = PageIndexRAG(llm_kit=my_kit)
    rag.ingest("report.pdf")          # page-by-page PDF indexing
    rag.ingest("notes.txt")           # sliding-window text indexing

    # Retrieve without LLM
    results = rag.retrieve("revenue growth", top_k=5)

    # Full RAG: retrieve + generate
    response = rag.query("What were the Q3 revenue figures?")
    print(response.answer.content)
"""

from ractogateway.rag.page_index._models import PageEntry, PageIndexResponse, PageIndexResult
from ractogateway.rag.page_index.pipeline import PageIndexRAG

__all__ = [
    "PageIndexRAG",
    "PageEntry",
    "PageIndexResult",
    "PageIndexResponse",
]
