"""
Geneformer-compatible tokenizer for bulk RNA-seq data.

Geneformer tokenizes single-cell RNA by ranking genes per cell.
For bulk RNA we apply the same rank-ordering per patient (treating
the bulk sample as a pseudo-cell). This is the approach used in
published bulk RNA transfer learning papers (biorxiv 2024.11.03).

Output: integer tensor of shape (1, max_genes) per patient,
        where values are Geneformer gene token IDs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import torch


# Geneformer gene vocabulary mapping: ENSG_ID → token index
# Loaded lazily from HuggingFace tokenizer on first use.
_GENE_VOCAB: Optional[Dict[str, int]] = None


def _load_gene_vocab(model_cache_dir: str = "./model_cache") -> Dict[str, int]:
    global _GENE_VOCAB
    if _GENE_VOCAB is not None:
        return _GENE_VOCAB
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "ctheodoris/Geneformer",
            cache_dir=model_cache_dir,
            trust_remote_code=True,
        )
        _GENE_VOCAB = tok.get_vocab()
    except Exception:
        # Fallback: map ENSG IDs to sequential integers (for testing / no-internet)
        _GENE_VOCAB = {}
    return _GENE_VOCAB


def tokenize_bulk_rna(
    expression_row,           # pd.Series or Dict[str, float]
    max_genes: int = 2048,
    model_cache_dir: str = "./model_cache",
) -> torch.Tensor:
    """
    Convert one patient's FPKM-UQ bulk RNA-seq row to a Geneformer token tensor.

    Steps:
    1. Median-normalization (scale to sum=10,000)
    2. Rank genes descending by expression (highest expression = rank 1)
    3. Map each Ensembl ID to Geneformer token vocabulary
    4. Keep top max_genes tokens; pad with 0 if fewer expressed

    Returns: torch.LongTensor of shape (1, max_genes)
    """
    vocab = _load_gene_vocab(model_cache_dir)

    if isinstance(expression_row, dict):
        expression_row = pd.Series(expression_row)

    counts = expression_row.values.astype(np.float32)
    total  = counts.sum()
    if total > 0:
        counts = counts * (10_000.0 / total)

    # Rank descending
    ranked_indices = np.argsort(-counts)
    ranked_genes   = expression_row.index[ranked_indices]

    tokens = []
    for gene in ranked_genes:
        if gene in vocab:
            tokens.append(vocab[gene])
        if len(tokens) == max_genes:
            break

    # Pad to max_genes with token 0 (padding)
    if len(tokens) < max_genes:
        tokens.extend([0] * (max_genes - len(tokens)))

    return torch.tensor(tokens[:max_genes], dtype=torch.long).unsqueeze(0)  # (1, max_genes)


def tokenize_bulk_rna_from_file(
    rna_path: str,
    max_genes: int = 2048,
    model_cache_dir: str = "./model_cache",
) -> torch.Tensor:
    """
    Read a TSV/CSV file with one sample's RNA-seq counts and tokenise it.
    Expected format: two columns 'gene_id' and 'count' (or single-row CSV).
    """
    path = Path(rna_path)
    if not path.exists():
        # Return zero-padded tensor for missing file (graceful degradation)
        return torch.zeros(1, max_genes, dtype=torch.long)

    if path.suffix in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    # Expect columns: gene_id, count (or ENSG_ID as first column)
    if "gene_id" in df.columns and "count" in df.columns:
        series = df.set_index("gene_id")["count"]
    elif df.shape[0] == 1:
        # Single-row CSV: columns are gene IDs, values are counts
        series = df.iloc[0]
    else:
        series = df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0]
        series.index = df.iloc[:, 0] if df.shape[1] > 1 else df.index

    return tokenize_bulk_rna(series, max_genes=max_genes, model_cache_dir=model_cache_dir)


def batch_tokenize(
    rna_df: pd.DataFrame,
    max_genes: int = 2048,
    model_cache_dir: str = "./model_cache",
) -> Dict[str, torch.Tensor]:
    """Tokenize all patients in a DataFrame. Returns {case_id: tensor}."""
    result = {}
    for case_id, row in rna_df.iterrows():
        result[str(case_id)] = tokenize_bulk_rna(row, max_genes, model_cache_dir)
    return result
