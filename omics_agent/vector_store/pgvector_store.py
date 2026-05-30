"""
pgvector-backed patient store — swappable alternative to FAISS.

Activated via VECTOR_BACKEND=pgvector in .env.
Requires: psycopg2-binary, pgvector extension in Postgres.

Table schema (auto-created):
    patients_vec(id SERIAL, patient_id TEXT UNIQUE, embedding VECTOR(256),
                 survival_months FLOAT, event INT, stage INT, subtype TEXT,
                 clinical_text TEXT)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS patients_vec (
    id              SERIAL PRIMARY KEY,
    patient_id      TEXT UNIQUE NOT NULL,
    embedding       VECTOR({dim}),
    survival_months FLOAT,
    event           INTEGER,
    stage           INTEGER,
    subtype         TEXT,
    clinical_text   TEXT
);
CREATE INDEX IF NOT EXISTS patients_vec_embedding_idx
    ON patients_vec USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


class PgVectorStore:
    """
    pgvector backend for patient similarity search.

    Parameters
    ----------
    connection_string : PostgreSQL DSN, e.g.
        "postgresql://user:pass@localhost:5432/omicsdb"
    dim               : embedding dimension (default 256)
    """

    def __init__(self, connection_string: str, dim: int = 256):
        self.connection_string = connection_string
        self.dim = dim
        self._conn = None

    # ------------------------------------------------------------------
    def connect(self) -> None:
        import psycopg2
        from pgvector.psycopg2 import register_vector
        self._conn = psycopg2.connect(self.connection_string)
        register_vector(self._conn)
        with self._conn.cursor() as cur:
            sql = _CREATE_TABLE.format(dim=self.dim)
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
        self._conn.commit()

    def _ensure_connected(self) -> None:
        if self._conn is None:
            self.connect()

    # ------------------------------------------------------------------
    def upsert(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        """Insert or update patients. embeddings shape: (N, dim)."""
        self._ensure_connected()
        emb = np.array(embeddings, dtype=np.float32)
        # L2-normalise for cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-10)

        with self._conn.cursor() as cur:
            for i, m in enumerate(metadata):
                cur.execute(
                    """
                    INSERT INTO patients_vec
                        (patient_id, embedding, survival_months, event, stage, subtype, clinical_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (patient_id) DO UPDATE SET
                        embedding       = EXCLUDED.embedding,
                        survival_months = EXCLUDED.survival_months,
                        event           = EXCLUDED.event,
                        stage           = EXCLUDED.stage,
                        subtype         = EXCLUDED.subtype,
                        clinical_text   = EXCLUDED.clinical_text
                    """,
                    (
                        m.get("patient_id", f"p{i}"),
                        emb[i].tolist(),
                        float(m.get("survival_months", 0.0)),
                        int(m.get("event", 0)),
                        int(m.get("stage", 0)),
                        str(m.get("subtype", "")),
                        str(m.get("clinical_text", "")),
                    ),
                )
        self._conn.commit()

    # ------------------------------------------------------------------
    def search(
        self,
        query: np.ndarray,     # (dim,)
        k: int = 20,
        max_stage_diff: Optional[int] = 1,
        query_stage: Optional[int] = None,
    ) -> List[dict]:
        """
        Return up to k nearest patients by cosine similarity.
        Applies stage hard-filter if query_stage is provided.
        """
        self._ensure_connected()
        q = np.array(query, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-10)

        stage_clause = ""
        params: list = [q.tolist(), k]
        if query_stage is not None and max_stage_diff is not None:
            stage_clause = f"WHERE ABS(stage - {int(query_stage)}) <= {int(max_stage_diff)}"

        sql = f"""
            SELECT patient_id, survival_months, event, stage, subtype, clinical_text,
                   1 - (embedding <=> %s::vector) AS cosine_score
            FROM patients_vec
            {stage_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [q.tolist(), q.tolist(), k]

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        cols = ["patient_id", "survival_months", "event", "stage",
                "subtype", "clinical_text", "faiss_score"]
        return [dict(zip(cols, r)) for r in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
