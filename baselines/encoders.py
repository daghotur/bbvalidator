"""
baselines/encoders.py
---------------------
Два базлайновых энкодера для сравнения с HybridProteinEncoder.

Принцип честного сравнения: фронтенд (биофизические признаки + PCA),
пулинг, головы, лосс и протокол обучения одинаковы у всех моделей —
меняется только энкодер. Поэтому разница в метриках атрибутируется
именно архитектуре энкодера.

Контракт совпадает с HybridProteinEncoder:
  вход  — dict от BiophysicalFrontend (node_feats, edge_indices, edge_attrs, mask)
  выход — node embeddings [B, N, d_model]
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, TransformerConv


class BaselineMLPEncoder(nn.Module):
    """Независимый MLP на каждый остаток: никакого обмена информацией
    между позициями (ни графа, ни внимания)."""

    def __init__(self, node_in_dim: int = 31, d_model: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: dict) -> torch.Tensor:
        x = self.net(features["node_feats"])  # [B, N, d_model]
        return x * features["mask"].unsqueeze(-1)


class BaselineGPSEncoder(nn.Module):
    """Стек GPSConv (TransformerConv + глобальное внимание) на kNN-графе
    фронтенда — аналог main2.py, пересаженный на общий контракт."""

    def __init__(
        self,
        node_in_dim: int = 31,
        pair_in_dim: int = 29,  # PairFeatureBuilder.feature_dim
        d_model: int = 128,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.pair_in_dim = pair_in_dim
        self.node_proj = nn.Linear(node_in_dim, d_model)

        self.gps_layers = nn.ModuleList([
            GPSConv(
                channels=d_model,
                conv=TransformerConv(
                    in_channels=d_model,
                    out_channels=d_model // heads,
                    heads=heads,
                    concat=True,
                    beta=True,
                    dropout=dropout,
                    edge_dim=pair_in_dim,
                ),
                heads=heads,
                dropout=dropout,
                attn_type="multihead",
                norm="batch_norm",
            )
            for _ in range(num_layers)
        ])

    def forward(self, features: dict) -> torch.Tensor:
        x_raw = features["node_feats"]  # [B, N, node_in_dim]
        mask = features["mask"]
        e_idx = features["edge_index"]  # [2, E] — упакованная нумерация
        e_attr = features["edge_attr"]  # [E, pair_in_dim]

        B, N, _ = x_raw.shape
        device = x_raw.device

        x = self.node_proj(x_raw)

        # Упаковка: убираем паддинг, стыкуем графы батча в один (как в гибриде).
        # x[mask] даёт ту же нумерацию, в которой фронтенд выдал рёбра.
        x_packed = x[mask]
        batch_vec = (
            torch.arange(B, device=device).unsqueeze(1).expand(B, N)[mask]
        )

        for layer in self.gps_layers:
            x_packed = layer(x_packed, edge_index=e_idx, batch=batch_vec, edge_attr=e_attr)

        # Под autocast batch_norm внутри GPSConv остаётся в fp32, поэтому
        # dtype x_packed может отличаться от x — буфер создаём по x_packed.
        x_out = torch.zeros(
            (B, N, x_packed.size(-1)), device=device, dtype=x_packed.dtype
        )
        x_out[mask] = x_packed
        return x_out
