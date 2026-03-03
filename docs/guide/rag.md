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
