"""
Hybrid FAISS + BM25 retriever with stage-proximity hard filter.

Retrieval score = 0.7 × faiss_cosine + 0.3 × bm25_norm
Hard filter: |stage_query - stage_candidate| > 1 → discard
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .indexer import PatientIndexer


def _bm25_score(
    query_tokens: List[str],
    doc_tokens:   List[str],
    avg_dl:       float = 10.0,
    k1:           float = 1.5,
    b:            float = 0.75,
) -> float:
    """Single-document BM25 score (no IDF — uniform term weight for clinical text)."""
    if not query_tokens or not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    freq: dict[str, int] = {}
    for tok in doc_tokens:
        freq[tok] = freq.get(tok, 0) + 1
    score = 0.0
    for tok in set(query_tokens):
        f = freq.get(tok, 0)
        score += (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avg_dl, 1)))
    return score


class PatientRetriever:
    """
    Hybrid retriever: FAISS cosine similarity + BM25 on clinical text tokens.

    Parameters
    ----------
    indexer     : Loaded PatientIndexer
    faiss_weight: Weight for cosine similarity (default 0.7)
    bm25_weight : Weight for BM25 (default 0.3)
    max_stage_diff: Hard filter — discard if |stage_q - stage_c| > this (default 1)
    """

    def __init__(
        self,
        indexer: PatientIndexer,
        faiss_weight:    float = 0.7,
        bm25_weight:     float = 0.3,
        max_stage_diff:  int   = 1,
    ):
        self.indexer         = indexer
        self.faiss_weight    = faiss_weight
        self.bm25_weight     = bm25_weight
        self.max_stage_diff  = max_stage_diff

    def retrieve(
        self,
        query_embedding:   np.ndarray,    # (256,)
        query_stage:       int,
        query_clinical_text: str,
        top_k:             int = 10,
        candidate_k:       int = 50,
    ) -> List[dict]:
        """
        Return up to top_k similar patients after hybrid reranking.

        Each returned dict has:
            patient_id, survival_months, event, stage, subtype,
            faiss_score, bm25_score, hybrid_score, rank
        """
        # 1. FAISS top-candidate_k retrieval
        faiss_scores, faiss_indices = self.indexer.search(query_embedding, k=candidate_k)
        if len(faiss_indices) == 0:
            return []

        metadata = self.indexer.get_metadata(faiss_indices)
        meta_by_idx = {m["faiss_idx"]: m for m in metadata}

        query_tokens = query_clinical_text.lower().split()

        candidates = []
        for i, fidx in enumerate(faiss_indices):
            if fidx < 0:
                continue
            m = meta_by_idx.get(int(fidx))
            if m is None:
                continue

            # Hard stage filter
            stage_diff = abs(query_stage - int(m.get("stage") or 0))
            if stage_diff > self.max_stage_diff:
                continue

            doc_tokens = (m.get("clinical_text") or "").lower().split()
            bm25 = _bm25_score(query_tokens, doc_tokens)
            candidates.append({**m, "_faiss": float(faiss_scores[i]), "_bm25": bm25})

        if not candidates:
            return []

        # 2. Normalise BM25 scores to [0, 1]
        bm25_vals = [c["_bm25"] for c in candidates]
        bm25_max  = max(bm25_vals) if max(bm25_vals) > 0 else 1.0
        bm25_norm = [v / bm25_max for v in bm25_vals]

        # 3. Hybrid score + sort
        for c, b_norm in zip(candidates, bm25_norm):
            c["faiss_score"]  = c.pop("_faiss")
            c["bm25_score"]   = c.pop("_bm25")
            c["hybrid_score"] = (
                self.faiss_weight * c["faiss_score"]
                + self.bm25_weight * b_norm
            )

        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top = candidates[:top_k]
        for rank, c in enumerate(top, 1):
            c["rank"] = rank

        return top

    def cohort_stats(self, patients: List[dict]) -> dict:
        """Summarise retrieved cohort: median survival, event rate, n."""
        if not patients:
            return {"n": 0, "median_survival_months": None, "event_rate": None}
        surv = [p["survival_months"] for p in patients if p.get("survival_months") is not None]
        events = [p["event"] for p in patients if p.get("event") is not None]
        median_surv = float(np.median(surv)) if surv else 0.0
        event_rate  = float(np.mean(events)) if events else 0.0
        return {
            "n":                      len(patients),
            "median_survival_months": round(median_surv, 1),
            "event_rate":             round(event_rate, 3),
        }
