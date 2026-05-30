"""
LoRA fine-tuning of Geneformer on TCGA survival data.

Adds a regression head predicting log-hazard ratio and fine-tunes
only the last 2 transformer layers + projection head using PEFT LoRA adapters.

This script is optional — the model works with frozen Geneformer embeddings.
Run via: python scripts/run_finetune.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


class GeneformerSurvivalHead(nn.Module):
    """Thin head on top of Geneformer [CLS] → log-hazard ratio."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        return self.head(cls_emb).squeeze(-1)   # (B,)


def _breslow_loss(risk_scores: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Breslow approximation of Cox partial log-likelihood."""
    order = torch.argsort(durations, descending=True)
    risk = risk_scores[order]
    ev   = events[order].float()
    log_cs = torch.logcumsumexp(risk, dim=0)
    return -torch.mean((risk - log_cs) * ev)


def finetune_geneformer(
    model,
    train_tokens: torch.Tensor,    # (N, max_genes)
    train_durations: np.ndarray,
    train_events: np.ndarray,
    n_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu",
    save_dir: Optional[str] = None,
) -> GeneformerSurvivalHead:
    """
    Fine-tune the last 2 Geneformer layers + a survival head.

    model: the loaded GeneformerEncoder._model (HF model)
    Returns the trained GeneformerSurvivalHead (head only — base model PEFT adapters saved separately).
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["query", "value"],
            modules_to_save=["encoder.layer.4", "encoder.layer.5"],
        )
        peft_model = get_peft_model(model, lora_config)
    except ImportError:
        peft_model = model

    head = GeneformerSurvivalHead(hidden_size=256).to(device)
    optimizer = torch.optim.AdamW(
        list(peft_model.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    dur_t = torch.tensor(train_durations, dtype=torch.float32).to(device)
    ev_t  = torch.tensor(train_events,    dtype=torch.float32).to(device)

    for epoch in range(n_epochs):
        peft_model.train(); head.train()
        out = peft_model(input_ids=train_tokens.to(device))
        cls_emb = out.last_hidden_state[:, 0, :]
        risk    = head(cls_emb)
        loss    = _breslow_loss(risk, dur_t, ev_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), f"{save_dir}/survival_head.pt")
        try:
            peft_model.save_pretrained(save_dir)
        except Exception:
            pass

    return head
