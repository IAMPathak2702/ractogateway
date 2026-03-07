"""PageIndexRAG — vectorless, page-level BM25 retrieval.

Quick start::

    from ractogateway.rag.page_index import PageIndexRAG

    rag = PageIndexRAG(llm_kit=kit)
    rag.ingest("report.pdf")
    rag.ingest("notes.txt")

    # Retrieve only
    results = rag.retrieve("revenue growth", top_k=5)

    # Full RAG: retrieve + generate
    response = rag.query("What were the Q3 revenue figures?")
    print(response.answer.content)
"""

from ractogateway.rag.page_index._models import (
    PageEntry,
    PageIndexResponse,
    PageIndexResult,
)
from ractogateway.rag.page_index._ocr import (
    AWSTextractBackend,
    AzureDocumentIntelligenceBackend,
    BaseOcrBackend,
    EasyOcrBackend,
    GoogleDocumentAIBackend,
    GoogleVisionBackend,
    TesseractOcrBackend,
)
from ractogateway.rag.page_index.pipeline import PageIndexRAG

__all__ = [
    "PageIndexRAG",
    "PageEntry",
    "PageIndexResult",
    "PageIndexResponse",
    # OCR backends
    "BaseOcrBackend",
    "TesseractOcrBackend",
    "EasyOcrBackend",
    "GoogleVisionBackend",
    "GoogleDocumentAIBackend",
    "AWSTextractBackend",
    "AzureDocumentIntelligenceBackend",
]
