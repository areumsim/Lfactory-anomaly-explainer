"""Anomaly Transformer for time series anomaly detection.

Simplified implementation of:
  Xu et al. (2022) "Anomaly Transformer: Time Series Anomaly Detection
  with Association Discrepancy" (ICLR 2022)

Core idea: learns prior-association (learnable Gaussian) and series-association
(self-attention) distributions. Anomalies show high association discrepancy
between the two.

Exports:
- detect_anomaly_transformer(series, ...) -> dict with scores, preds, params

Dependencies:
- torch (PyTorch)
"""
from __future__ import annotations

import math
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        self._mask = mask.unsqueeze(0).expand(B, -1, -1)

    @property
    def mask(self):
        return self._mask


class _AnomalyAttention(nn.Module):
    """Anomaly attention with prior-association and series-association."""

    def __init__(self, d_model, n_heads, d_keys=None, attn_dropout=0.0):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.scale = d_keys ** -0.5

        self.W_Q = nn.Linear(d_model, d_keys * n_heads)
        self.W_K = nn.Linear(d_model, d_keys * n_heads)
        self.W_V = nn.Linear(d_model, d_keys * n_heads)
        self.sigma = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.out_proj = nn.Linear(d_keys * n_heads, d_model)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads

        Q = self.W_Q(x).view(B, L, H, self.d_keys).transpose(1, 2)
        K = self.W_K(x).view(B, L, H, self.d_keys).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, self.d_keys).transpose(1, 2)

        # Series-association (standard attention)
        series_attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        series_attn = self.dropout(series_attn)

        # Prior-association (learnable Gaussian kernel)
        sigma = self.sigma.clamp(min=1e-4)
        distances = torch.abs(
            torch.arange(L, device=x.device).float().unsqueeze(0) -
            torch.arange(L, device=x.device).float().unsqueeze(1)
        )
        prior_attn = torch.softmax(
            -distances.unsqueeze(0).unsqueeze(0) / (2 * sigma ** 2 + 1e-8),
            dim=-1
        ).expand(B, H, L, L)

        out = (series_attn @ V).transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_proj(out)

        return out, series_attn, prior_attn


class _AnomalyTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.attn = _AnomalyAttention(d_model, n_heads, attn_dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, series_attn, prior_attn = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, series_attn, prior_attn


class _AnomalyTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2,
                 seq_len=100, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            _AnomalyTransformerLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (B, L, input_dim)
        B, L, _ = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :L, :]

        all_series, all_prior = [], []
        for layer in self.layers:
            h, s_attn, p_attn = layer(h)
            all_series.append(s_attn)
            all_prior.append(p_attn)

        recon = self.output_proj(h)
        return recon, all_series, all_prior


def _association_discrepancy(series_attns, prior_attns):
    """Compute KL-based association discrepancy per timestep."""
    disc = 0.0
    for s_attn, p_attn in zip(series_attns, prior_attns):
        # Symmetrized KL: KL(P||S) + KL(S||P)
        kl_ps = (p_attn * (torch.log(p_attn + 1e-8) - torch.log(s_attn + 1e-8))).sum(dim=-1)
        kl_sp = (s_attn * (torch.log(s_attn + 1e-8) - torch.log(p_attn + 1e-8))).sum(dim=-1)
        disc = disc + (kl_ps + kl_sp).mean(dim=1)  # average over heads -> (B, L)
    return disc / len(series_attns)


def detect_anomaly_transformer(
    series: List[float],
    seq_len: int = 100,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    quantile: float = 0.99,
    lambda_assoc: float = 3.0,
    random_state: int = 42,
    device: str = "auto",
    **kwargs,
) -> Dict[str, Any]:
    """Detect anomalies using Anomaly Transformer.

    Args:
        series: Input time series (univariate)
        seq_len: Sequence length for transformer input
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        epochs: Training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        quantile: Threshold quantile
        lambda_assoc: Weight for association discrepancy in loss
        random_state: Random seed
        device: "cpu", "cuda", or "auto"
    """
    import numpy as np

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n = len(series)
    arr = np.array(series, dtype=np.float32)

    # Normalize
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-8:
        sigma = 1.0
    arr_norm = (arr - mu) / sigma

    # Create sliding windows
    if n < seq_len + 1:
        seq_len = max(n // 2, 10)

    windows = []
    for i in range(n - seq_len + 1):
        windows.append(arr_norm[i:i + seq_len])
    windows = np.stack(windows)  # (N_windows, seq_len)
    X = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)  # (N, L, 1)

    # Build model
    model = _AnomalyTransformer(
        input_dim=1, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, seq_len=seq_len, dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training phase (minimax: minimize recon + maximize association discrepancy)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon, series_attns, prior_attns = model(batch_x)

            recon_loss = F.mse_loss(recon, batch_x)
            assoc_disc = _association_discrepancy(series_attns, prior_attns).mean()

            # Phase 1: minimize recon - lambda * assoc_disc
            loss = recon_loss - lambda_assoc * assoc_disc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

    # Inference: compute anomaly scores
    model.eval()
    all_scores = torch.zeros(len(windows), seq_len)

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = X[i:i + batch_size].to(device)
            recon, series_attns, prior_attns = model(batch)

            # Anomaly score = reconstruction error * association discrepancy
            recon_err = (recon - batch).pow(2).squeeze(-1)  # (B, L)
            assoc_disc = _association_discrepancy(series_attns, prior_attns)  # (B, L)

            # Combined score
            score = recon_err * (1 + assoc_disc)
            all_scores[i:i + batch.shape[0]] = score.cpu()

    # Aggregate overlapping windows: average scores per point
    point_scores = np.zeros(n, dtype=np.float64)
    point_counts = np.zeros(n, dtype=np.float64)
    for i in range(len(windows)):
        for j in range(seq_len):
            point_scores[i + j] += all_scores[i, j].item()
            point_counts[i + j] += 1
    point_counts = np.maximum(point_counts, 1)
    point_scores = point_scores / point_counts

    # Normalize to [0, 1]
    s_min, s_max = point_scores.min(), point_scores.max()
    if s_max - s_min > 1e-12:
        point_scores = (point_scores - s_min) / (s_max - s_min)

    # Threshold
    sorted_scores = np.sort(point_scores)
    thr_idx = min(int(quantile * len(sorted_scores)), len(sorted_scores) - 1)
    threshold = sorted_scores[thr_idx]
    preds = [1 if s >= threshold else 0 for s in point_scores]

    return {
        "scores": point_scores.tolist(),
        "preds": preds,
        "params": {
            "method": "anomaly_transformer",
            "seq_len": seq_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "epochs": epochs,
            "lambda_assoc": lambda_assoc,
            "quantile": quantile,
            "threshold": float(threshold),
        },
    }
