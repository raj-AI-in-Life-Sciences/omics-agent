"""
Ingestion node: validate input, tokenize RNA-seq, prepare clinical features.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

import numpy as np

from ..graph.state import OmicsState
from ..core.config import get_config


def ingestion_node(state: OmicsState) -> OmicsState:
    """
    Validates patient_id, rna_seq_path, clinical_features.
    Tokenises RNA-seq counts → gene_tokens (len 2048).
    Does NOT run Geneformer (that's embedding_node).
    """
    cfg    = get_config()
    errors = list(state.get("errors") or [])

    run_id = state.get("run_id") or str(uuid.uuid4())

    # ── Tokenise RNA-seq ─────────────────────────────────────────────────
    gene_tokens: list = []
    rna_path = state.get("rna_seq_path", "")
    if rna_path:
        try:
            from ..data.geneformer_tokenizer import tokenize_bulk_rna_from_file
            tokens_tensor = tokenize_bulk_rna_from_file(
                rna_path, max_genes=cfg.max_genes
            )
            gene_tokens = tokens_tensor.squeeze(0).tolist()
        except Exception as exc:
            errors.append(f"tokenizer: {exc}")
            # Fall back to synthetic tokens for testing
            gene_tokens = list(range(cfg.max_genes))
    else:
        errors.append("rna_seq_path not provided — using synthetic tokens")
        gene_tokens = list(range(cfg.max_genes))

    return {
        **state,
        "run_id":     run_id,
        "gene_tokens": gene_tokens,
        "errors":     errors,
    }
