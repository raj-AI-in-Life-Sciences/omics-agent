"""
Geneformer encoder for OmicsAgent.

Loads the Geneformer model from HuggingFace, runs a forward pass on
tokenized bulk RNA, and returns the [CLS] hidden state as a 256-d embedding.

Embeddings are persisted to HDF5 cache (see cache.py) to avoid recomputation.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional


class GeneformerEncoder:
    """
    Wraps Geneformer for [CLS] embedding extraction.

    Usage:
        enc = GeneformerEncoder()
        embedding = enc.encode(token_tensor)   # (1, 2048) → (256,)
    """

    def __init__(
        self,
        model_name: str = "ctheodoris/Geneformer",
        device: str = "cpu",
        cache_dir: str = "./model_cache",
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self._model = None
        self._config = None
        self.cache_dir = cache_dir

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoConfig
        self._config = AutoConfig.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        ).to(self.device).eval()
        # Freeze all parameters (embeddings used as features, not fine-tuned here)
        for p in self._model.parameters():
            p.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        self._load()
        return self._config.hidden_size   # 256 for Geneformer 6L-30M

    @torch.no_grad()
    def encode(self, token_tensor: torch.Tensor) -> np.ndarray:
        """
        token_tensor: (1, max_genes) LongTensor
        Returns: numpy array (hidden_size,) — the [CLS] representation
        """
        self._load()
        token_tensor = token_tensor.to(self.device)
        outputs = self._model(input_ids=token_tensor)
        cls_emb = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        return cls_emb.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, token_batch: torch.Tensor) -> np.ndarray:
        """
        token_batch: (B, max_genes) LongTensor
        Returns: (B, hidden_size) float32 array
        """
        self._load()
        token_batch = token_batch.to(self.device)
        outputs = self._model(input_ids=token_batch)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb.cpu().numpy().astype(np.float32)
