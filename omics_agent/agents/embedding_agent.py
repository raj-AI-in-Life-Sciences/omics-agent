"""
Embedding node: Geneformer forward pass → [CLS] embedding → HDF5 cache.
"""

from __future__ import annotations

import numpy as np
import torch

from ..graph.state import OmicsState
from ..core.config import get_config


def embedding_node(state: OmicsState) -> OmicsState:
    """
    Checks HDF5 cache first; runs Geneformer if not cached.
    Returns geneformer_embedding: list[float] length 256.
    """
    cfg    = get_config()
    errors = list(state.get("errors") or [])
    pid    = state.get("patient_id", "unknown")

    embedding: list = []

    try:
        from ..embeddings.cache import EmbeddingCache
        cache = EmbeddingCache(cfg.embedding_cache_path)

        if cache.contains(pid):
            embedding = cache.load(pid).tolist()
        else:
            gene_tokens = state.get("gene_tokens", [])
            if gene_tokens:
                from ..embeddings.geneformer_encoder import GeneformerEncoder
                encoder = GeneformerEncoder(
                    model_name=cfg.geneformer_model_name,
                    embedding_dim=cfg.geneformer_embedding_dim,
                )
                token_tensor = torch.tensor(
                    gene_tokens, dtype=torch.long
                ).unsqueeze(0)
                emb = encoder.encode(token_tensor)
                cache.save(pid, emb)
                embedding = emb.tolist()
            else:
                errors.append("embedding_node: no gene_tokens — using zero embedding")
                embedding = [0.0] * cfg.geneformer_embedding_dim
    except Exception as exc:
        errors.append(f"embedding_node: {exc}")
        embedding = [0.0] * cfg.geneformer_embedding_dim

    return {
        **state,
        "geneformer_embedding": embedding,
        "errors": errors,
    }
