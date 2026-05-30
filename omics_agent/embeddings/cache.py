"""
HDF5-backed embedding cache for OmicsAgent.

Avoids repeated Geneformer inference by persisting [CLS] embeddings.
Embeddings are stored as float16 to reduce disk footprint (~256 dims × 2 bytes per patient).
"""

from __future__ import annotations

import numpy as np
import h5py
from pathlib import Path
from typing import Optional


class EmbeddingCache:
    """
    Read/write HDF5 cache for patient embeddings.

    Usage:
        cache = EmbeddingCache("data/processed/embeddings.h5")
        cache.save("TCGA-A1-A0SD", emb)
        emb = cache.load("TCGA-A1-A0SD")     # None if not cached
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, patient_id: str, embedding: np.ndarray) -> None:
        with h5py.File(self.path, "a") as f:
            key = patient_id.replace("/", "_")
            if key in f:
                del f[key]
            f.create_dataset(key, data=embedding.astype(np.float16), compression="gzip")

    def load(self, patient_id: str) -> Optional[np.ndarray]:
        if not self.path.exists():
            return None
        key = patient_id.replace("/", "_")
        with h5py.File(self.path, "r") as f:
            if key not in f:
                return None
            return f[key][:].astype(np.float32)

    def load_all(self) -> dict[str, np.ndarray]:
        if not self.path.exists():
            return {}
        result = {}
        with h5py.File(self.path, "r") as f:
            for key in f.keys():
                result[key] = f[key][:].astype(np.float32)
        return result

    def contains(self, patient_id: str) -> bool:
        if not self.path.exists():
            return False
        key = patient_id.replace("/", "_")
        with h5py.File(self.path, "r") as f:
            return key in f

    def __len__(self) -> int:
        if not self.path.exists():
            return 0
        with h5py.File(self.path, "r") as f:
            return len(f.keys())
