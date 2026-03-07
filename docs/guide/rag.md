# RAG — Retrieval-Augmented Generation

RactoGateway ships two complementary RAG pipelines:

| Pipeline | Requires embeddings | Requires vector store | Best for |
| --- | :---: | :---: | --- |
| `RactoRAG` | Yes | Yes | Semantic / conceptual queries |
| `PageIndexRAG` | **No** | **No** | Keyword-rich exact-term queries, cost-sensitive setups |

---

## RactoRAG

`RactoRAG` provides a full pipeline: read → chunk → process → embed → store → retrieve.

---

## PageIndexRAG — Vectorless BM25 RAG

`PageIndexRAG` indexes documents at the **page level** and retrieves using a two-stage
decision-tree approach — no embedding API calls, no external vector store required.

### How it works

1. **Decision index (routing):** Each page's top-N TF-weighted keywords are stored in an
   inverted index (`term → page IDs`). A query is tokenised and the index returns the union
   of matching page IDs in O(|query terms|) time.
2. **BM25 scoring:** Only the candidate pages from step 1 are scored with Okapi BM25
   (k1=1.5, b=0.75), giving accurate relevance ordering without scanning the full corpus.

### Quick start

```python
from ractogateway.rag.page_index import PageIndexRAG
from ractogateway import openai_developer_kit as gpt

kit = gpt.Chat(model="gpt-4o", default_prompt=my_prompt)
rag = PageIndexRAG(llm_kit=kit)

rag.ingest("report.pdf")           # page-by-page via pypdf
rag.ingest("notes.txt")            # sliding-window (1 000 chars, 100 overlap)
rag.ingest_text("raw text...", source="memo")

# Retrieve-only (no LLM needed)
results = rag.retrieve("Q3 revenue APAC", top_k=5)
for r in results:
    print(r.rank, r.score, r.entry.source, r.entry.page_number, r.matched_terms)

# Full RAG: retrieve + generate
response = rag.query("What were the Q3 APAC revenue figures?")
print(response.answer.content)

# Async
await rag.aingest("big_report.pdf")
results = await rag.aretrieve("revenue", top_k=3)
response = await rag.aquery("Summarise findings.")
```

### Page splitting strategy

| File type | Strategy |
| --- | --- |
| PDF (`.pdf`) | `pypdf` — one `PageEntry` per PDF page |
| All others | Sliding character windows (`page_size=1000`, `page_overlap=100`) |

### OCR support for scanned and handwritten PDFs

When a PDF page contains no embedded text (scanned document, handwritten notes,
image-only PDF), `pypdf` returns an empty string. Pass an `ocr_backend` to
automatically fall back to OCR for those pages:

```python
from ractogateway.rag.page_index import PageIndexRAG, TesseractOcrBackend

rag = PageIndexRAG(
    llm_kit=kit,
    ocr_backend=TesseractOcrBackend(lang="eng"),
    ocr_fallback=True,          # default: only OCR pages with no embedded text
    min_ocr_confidence=40.0,    # skip pages where mean word confidence < 40 (0–100)
)
rag.ingest("scanned_contract.pdf")   # digital pages use pypdf, blank pages use OCR
```

When `ocr_fallback=True` (default) only empty pages trigger OCR — digital pages are
never sent to the OCR backend, keeping costs low. Set `ocr_fallback=False` to force
OCR on every page regardless of embedded text.

OCR metadata is stored on every `PageEntry`:

```python
entries = rag.ingest("scanned.pdf")
for e in entries:
    print(e.ocr_applied, e.ocr_confidence)   # True  87.4
```

#### Available OCR backends

| Backend | Extra | Notes |
| --- | --- | --- |
| `TesseractOcrBackend` | `rag-ocr-tesseract` | Free, offline; requires Tesseract binary |
| `EasyOcrBackend` | `rag-ocr-easy` | Deep-learning, 80+ languages, offline |
| `GoogleVisionBackend` | `rag-ocr-google` | Google Cloud Vision `DOCUMENT_TEXT_DETECTION` |
| `GoogleDocumentAIBackend` | `rag-ocr-google` | Google Document AI; best for tables / forms |
| `AWSTextractBackend` | `rag-ocr-aws` | AWS Textract; great for key-value pairs |
| `AzureDocumentIntelligenceBackend` | `rag-ocr-azure` | Azure Form Recognizer v4 |

```bash
pip install ractogateway[rag-ocr-tesseract]   # Tesseract (free, offline)
pip install ractogateway[rag-ocr-easy]        # EasyOCR (deep-learning, offline)
pip install ractogateway[rag-ocr-google]      # Google Vision + Document AI
pip install ractogateway[rag-ocr-aws]         # AWS Textract
pip install ractogateway[rag-ocr-azure]       # Azure Document Intelligence
```

**Tesseract** (free, offline, no API key):

```python
from ractogateway.rag.page_index import TesseractOcrBackend

# Multi-language
backend = TesseractOcrBackend(lang="eng+deu", config="--psm 6")
rag = PageIndexRAG(llm_kit=kit, ocr_backend=backend, min_ocr_confidence=50.0)
```

**Google Document AI** (best for structured docs — invoices, contracts, forms):

```python
from ractogateway.rag.page_index import GoogleDocumentAIBackend

backend = GoogleDocumentAIBackend(
    project_id="my-gcp-project",
    processor_id="abc123def456",   # OCR or Form Parser processor
    location="us",
)
rag = PageIndexRAG(llm_kit=kit, ocr_backend=backend)
```

**AWS Textract**:

```python
from ractogateway.rag.page_index import AWSTextractBackend

backend = AWSTextractBackend(region_name="us-east-1")
# Credentials from env / ~/.aws/credentials / IAM role
rag = PageIndexRAG(llm_kit=kit, ocr_backend=backend)
```

**Azure Document Intelligence**:

```python
from ractogateway.rag.page_index import AzureDocumentIntelligenceBackend

backend = AzureDocumentIntelligenceBackend(
    endpoint="https://my-resource.cognitiveservices.azure.com/",
    api_key="...",
    model_id="prebuilt-read",   # or "prebuilt-document" for richer extraction
)
rag = PageIndexRAG(llm_kit=kit, ocr_backend=backend)
```

---

### Document deduplication

Re-ingesting the same file is a **no-op** — `PageIndexRAG` computes a SHA-256
hash of the raw file bytes on every `ingest()` call and returns the cached
entries immediately if the file was already indexed:

```python
entries_1 = rag.ingest("report.pdf")   # indexed — 12 pages
entries_2 = rag.ingest("report.pdf")   # no-op — returns same 12 entries instantly
assert entries_1 == entries_2
```

This prevents duplicate pages from inflating BM25 scores when the same file is
ingested from different paths or re-processed in a pipeline restart.

---

### Index persistence — save and load

Persist the full index (entries, BM25 weights, dedup hashes) to a JSON file
and reload it across process restarts:

```python
# Build and save
rag = PageIndexRAG(llm_kit=kit, ocr_backend=TesseractOcrBackend())
rag.ingest("report.pdf")
rag.save("./my_index.json")

# Reload in a new process — no re-ingestion needed
rag2 = PageIndexRAG.load("./my_index.json", llm_kit=kit)
response = rag2.query("What are the key findings?")
print(response.answer.content)
```

The saved JSON is human-readable and portable. Any `ocr_backend` configured at
save time must be re-supplied to `load()` if you intend to ingest new documents
after loading; it is not serialised.

---

### Removing documents

Remove all pages of a specific document from the index:

```python
entries = rag.ingest("old_report.pdf")
doc_id = entries[0].doc_id

removed = rag.remove_document(doc_id)
print(f"Removed {removed} pages")
```

`remove_document` also cleans up the BM25 term-frequency state and the
dedup hash, so the file can be re-ingested fresh afterwards.

---

### Parallel ingest for large corpora

**Async parallel** (`aingest_dir`) — up to `max_concurrent` files at once:

```python
import asyncio

async def main():
    entries = await rag.aingest_dir(
        "./docs/",
        pattern="**/*.pdf",
        max_concurrent=8,
        on_progress=lambda done, total: print(f"{done}/{total}"),
    )
    print(f"Indexed {len(entries)} pages")

asyncio.run(main())
```

**Sync with progress callback** (`ingest_dir`):

```python
def show_progress(done: int, total: int) -> None:
    print(f"[{done}/{total}] ingesting...")

entries = rag.ingest_dir("./docs/", on_progress=show_progress)
```

---

### Constructor parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `llm_kit` | `None` | Kit for generation; `None` = retrieve-only mode |
| `processors` | `[TextCleaner()]` | Text cleaning pipeline |
| `reader_registry` | Built-in | File reader registry |
| `context_template` | Built-in | `{context}` / `{question}` template |
| `default_prompt` | Built-in RAG prompt | Generation system prompt |
| `page_size` | `1000` | Max chars per window (non-PDF) |
| `page_overlap` | `100` | Char overlap between windows |
| `k1` | `1.5` | BM25 term-frequency saturation |
| `b` | `0.75` | BM25 length normalisation |
| `top_keywords` | `20` | Keywords per page in decision index |
| `ocr_backend` | `None` | OCR backend for scanned / handwritten PDFs |
| `ocr_fallback` | `True` | Only OCR pages with no embedded text |
| `min_ocr_confidence` | `0.0` | Drop OCR pages below this confidence (0–100) |

### Result models

**`PageEntry`** — one indexed page:

| Field | Type | Description |
| --- | --- | --- |
| `entry_id` | `str` | Auto UUID |
| `page_number` | `int \| None` | 1-based PDF page; `None` for windows |
| `content` | `str` | Post-processed page text |
| `source` | `str` | File path or label |
| `section_title` | `str \| None` | First Markdown heading on the page |
| `keywords` | `list[str]` | Top-N TF terms used by decision index |
| `doc_id` | `str` | Parent document UUID |
| `char_count` | `int` | `len(content)` |
| `ocr_applied` | `bool` | `True` when text was produced by OCR |
| `ocr_confidence` | `float \| None` | Mean word confidence (0–100); `None` if unavailable |

**`PageIndexResult`** — one retrieved page:

| Field | Type | Description |
| --- | --- | --- |
| `entry` | `PageEntry` | The retrieved page |
| `score` | `float` | BM25 relevance score |
| `rank` | `int` | 1-based rank |
| `matched_terms` | `list[str]` | Query tokens that hit this page |

**`PageIndexResponse`** — full query response:

| Field | Type | Description |
| --- | --- | --- |
| `answer` | `LLMResponse \| None` | Generated answer (`None` if no kit) |
| `sources` | `list[PageIndexResult]` | Retrieved pages |
| `query` | `str` | Original question |
| `context_used` | `str` | Context injected into LLM |
