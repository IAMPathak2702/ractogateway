# Installation

RactoGateway requires Python 3.10 or later.

## Core

```bash
pip install ractogateway
```

## LLM Providers

```bash
pip install ractogateway[openai]
pip install ractogateway[google]
pip install ractogateway[anthropic]
pip install ractogateway[all]        # all three providers
```

## RAG

```bash
pip install ractogateway[rag]        # readers + NLP (in-memory store)
pip install ractogateway[rag-all]    # everything RAG
```

Individual RAG extras:

| Extra | Installs |
|---|---|
| `rag-pdf` | `pypdf` |
| `rag-word` | `python-docx` |
| `rag-excel` | `openpyxl` |
| `rag-image` | `pillow` |
| `rag-nlp` | `nltk` |
| `rag-chroma` | `chromadb` |
| `rag-faiss` | `faiss-cpu`, `numpy` |
| `rag-pinecone` | `pinecone-client` |
| `rag-qdrant` | `qdrant-client` |
| `rag-weaviate` | `weaviate-client` |
| `rag-milvus` | `pymilvus` |
| `rag-pgvector` | `psycopg2-binary`, `pgvector` |
| `rag-voyage` | `voyageai` |
