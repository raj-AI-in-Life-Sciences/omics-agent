"""DeepSurv PyTorch survival model for OmicsAgent."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional


class DeepSurvNet(nn.Module):
    def __init__(self, input_dim: int = 267, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


def breslow_loss(risk: torch.Tensor, dur: torch.Tensor, ev: torch.Tensor) -> torch.Tensor:
    """Breslow approximation of negative log partial likelihood."""
    order = torch.argsort(dur, descending=True)
    risk  = risk[order]; ev = ev[order].float()
    return -torch.mean((risk - torch.logcumsumexp(risk, dim=0)) * ev)


class DeepSurvModel:
    """
    Wrapper around DeepSurvNet with training loop and numpy interface.
    """

    def __init__(
        self,
        input_dim: int = 267,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 50,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.net = DeepSurvNet(input_dim, dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.n_epochs = n_epochs
        self._feature_names: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        feature_names: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> "DeepSurvModel":
        X_t   = torch.tensor(X,         dtype=torch.float32).to(self.device)
        dur_t = torch.tensor(durations,  dtype=torch.float32).to(self.device)
        ev_t  = torch.tensor(events,     dtype=torch.float32).to(self.device)
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        self.net.train()
        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()
            risk = self.net(X_t)
            loss = breslow_loss(risk, dur_t, ev_t)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  DeepSurv epoch {epoch+1}/{self.n_epochs}  loss={loss.item():.4f}")
        return self

    @torch.no_grad()
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        return self.net(X_t).cpu().numpy().astype(np.float32)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
