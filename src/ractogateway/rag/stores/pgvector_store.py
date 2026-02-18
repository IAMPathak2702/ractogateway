"""PostgreSQL + pgvector store (lazy import).

Install with:  pip install ractogateway[rag-pgvector]
"""

from __future__ import annotations

import json
from typing import Any


def _require_pgvector() -> Any:
    try:
        import psycopg2  # noqa: PLC0415
        from pgvector.psycopg2 import register_vector  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PGVectorStore requires 'psycopg2-binary' and 'pgvector'. "
            "Install with:  pip install ractogateway[rag-pgvector]"
        ) from exc
    return psycopg2, register_vector


from ractogateway.rag._models.document import Chunk, ChunkMetadata
from ractogateway.rag._models.retrieval import RetrievalResult
from ractogateway.rag.stores.base import BaseVectorStore

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id  TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    content   TEXT NOT NULL,
    source    TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata  JSONB DEFAULT '{{}}',
    embedding vector({dim})
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS {table}_embedding_idx
ON {table} USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
"""


class PGVectorStore(BaseVectorStore):
    """Vector store backed by PostgreSQL with the pgvector extension.

    Parameters
    ----------
    dsn:
        PostgreSQL connection string (e.g.
        ``"postgresql://user:pass@localhost/mydb"``).
    table:
        Table name (default ``"rag_chunks"``).
    dimension:
        Embedding dimension.  Inferred on first add.
    distance:
        ``"cosine"``, ``"l2"``, or ``"inner"``.
    batch_size:
        Rows per INSERT batch.
    """

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "rag_chunks",
        dimension: int | None = None,
        distance: str = "cosine",
        batch_size: int = 100,
    ) -> None:
        self._dsn = dsn
        self._table = table
        self._dim = dimension
        self._distance = distance
        self._batch_size = batch_size
        self._conn: Any = None

    def _connect(self) -> Any:
        if self._conn is None or self._conn.closed:
            psycopg2, register_vector = _require_pgvector()
            self._conn = psycopg2.connect(self._dsn)
            register_vector(self._conn)
        return self._conn

    def _ensure_table(self, dim: int) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                _CREATE_TABLE.format(table=self._table, dim=dim)
            )
        conn.commit()

    def _dist_op(self) -> str:
        return {"cosine": "<=>", "l2": "<->", "inner": "<#>"}.get(self._distance, "<=>")

    def add(self, chunks: list[Chunk]) -> None:
        self._require_embeddings(chunks)
        dim = len(chunks[0].embedding)  # type: ignore[arg-type]
        if self._dim is None:
            self._dim = dim
        self._ensure_table(dim)
        conn = self._connect()
        with conn.cursor() as cur:
            for i in range(0, len(chunks), self._batch_size):
                batch = chunks[i : i + self._batch_size]
                cur.executemany(
                    f"""
                    INSERT INTO {self._table}
                        (chunk_id, doc_id, content, source, chunk_index, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            content   = EXCLUDED.content;
                    """,
                    [
                        (
                            c.chunk_id,
                            c.doc_id,
                            c.content,
                            c.metadata.source,
                            c.metadata.chunk_index,
                            json.dumps(c.metadata.extra),
                            c.embedding,
                        )
                        for c in batch
                    ],
                )
        conn.commit()

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if self._dim is None:
            return []
        conn = self._connect()
        op = self._dist_op()
        where_clauses: list[str] = []
        params: list[Any] = [embedding, top_k]
        if filters:
            for k, v in filters.items():
                where_clauses.append(f"metadata->>{k!r} = %s")
                params.insert(-1, str(v))

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"""
            SELECT chunk_id, doc_id, content, source, chunk_index, metadata,
                   embedding, embedding {op} %s AS distance
            FROM {self._table}
            {where_sql}
            ORDER BY distance
            LIMIT %s;
        """
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[RetrievalResult] = []
        for rank, row in enumerate(rows, start=1):
            cid, doc_id, content, source, chunk_index, meta, emb_raw, dist = row
            score = 1.0 - float(dist) if self._distance == "cosine" else -float(dist)
            chunk = Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                content=content,
                embedding=list(emb_raw) if emb_raw is not None else None,
                metadata=ChunkMetadata(
                    source=source,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_char=0,
                    end_char=len(content),
                    doc_id=doc_id,
                    extra=meta or {},
                ),
            )
            results.append(RetrievalResult(chunk=chunk, score=score, rank=rank))
        return results

    def delete(self, chunk_ids: list[str]) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE chunk_id = ANY(%s);",
                (chunk_ids,),
            )
        conn.commit()

    def clear(self) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self._table};")
        conn.commit()

    def count(self) -> int:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table};")
            row = cur.fetchone()
        return int(row[0]) if row else 0
