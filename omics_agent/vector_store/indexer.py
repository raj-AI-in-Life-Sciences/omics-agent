"""
FAISS IndexFlatIP patient indexer for OmicsAgent.

Stores L2-normalized Geneformer [CLS] embeddings so that dot product
equals cosine similarity. Metadata (patient_id, survival, event, stage,
subtype) lives in a companion SQLite table.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np


_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    faiss_idx       INTEGER PRIMARY KEY,
    patient_id      TEXT NOT NULL UNIQUE,
    survival_months REAL,
    event           INTEGER,
    stage           INTEGER,
    subtype         TEXT,
    clinical_text   TEXT
);
"""


class PatientIndexer:
    """
    Builds and persists a FAISS IndexFlatIP + SQLite metadata store.

    Usage:
        idx = PatientIndexer(index_path, db_path)
        idx.build(embeddings, metadata_list)  # write index + DB
        idx.load()                            # reload from disk
        vectors, ids = idx.search(query_vec, k=10)
    """

    def __init__(self, index_path: str, db_path: str):
        self.index_path = Path(index_path)
        self.db_path    = Path(db_path)
        self.index: Optional[faiss.IndexFlatIP] = None
        self._dim: int = 0

    # ------------------------------------------------------------------
    def build(
        self,
        embeddings: np.ndarray,        # (N, D) float32
        metadata:   List[dict],        # list of dicts with patient fields
    ) -> None:
        """L2-normalise, build FAISS index, persist index + SQLite."""
        emb = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb)

        self._dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(self._dim)
        self.index.add(emb)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        self._write_metadata(metadata)

    def load(self) -> None:
        """Load FAISS index from disk (SQLite is queried on demand)."""
        self.index = faiss.read_index(str(self.index_path))
        self._dim  = self.index.d

    # ------------------------------------------------------------------
    def search(
        self,
        query: np.ndarray,    # (D,) or (1, D)
        k: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (cosine_scores, faiss_indices) for top-k neighbours.
        query is L2-normalised in place before search.
        """
        q = np.array(query, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        return scores[0], indices[0]

    # ------------------------------------------------------------------
    def get_metadata(self, faiss_indices: np.ndarray) -> List[dict]:
        """Fetch metadata rows for a list of FAISS indices."""
        placeholders = ",".join("?" for _ in faiss_indices)
        con = sqlite3.connect(str(self.db_path))
        cur = con.execute(
            f"SELECT faiss_idx, patient_id, survival_months, event, stage, subtype, clinical_text "
            f"FROM patients WHERE faiss_idx IN ({placeholders})",
            faiss_indices.tolist(),
        )
        rows = cur.fetchall()
        con.close()
        cols = ["faiss_idx", "patient_id", "survival_months", "event",
                "stage", "subtype", "clinical_text"]
        return [dict(zip(cols, r)) for r in rows]

    # ------------------------------------------------------------------
    def _write_metadata(self, metadata: List[dict]) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db_path))
        con.execute(_SCHEMA)
        rows = []
        for i, m in enumerate(metadata):
            rows.append((
                i,
                m.get("patient_id", f"p{i}"),
                float(m.get("survival_months", 0.0)),
                int(m.get("event", 0)),
                int(m.get("stage", 0)),
                str(m.get("subtype", "")),
                str(m.get("clinical_text", "")),
            ))
        con.executemany(
            "INSERT OR REPLACE INTO patients "
            "(faiss_idx, patient_id, survival_months, event, stage, subtype, clinical_text) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        con.commit()
        con.close()

    @property
    def n_patients(self) -> int:
        return self.index.ntotal if self.index else 0
