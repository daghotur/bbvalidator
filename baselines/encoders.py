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
        pair_in_dim: int = 20,
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
        edge_indices = features["edge_indices"]
        edge_attrs = features["edge_attrs"]

        B, N, _ = x_raw.shape
        device = x_raw.device

        x = self.node_proj(x_raw)

        # Упаковка: убираем паддинг, стыкуем графы батча в один (как в гибриде)
        packed_nodes = []
        batch_ids = []
        global_edge_indices = []
        global_edge_attrs = []
        offset = 0

        for b in range(B):
            valid_x = x[b][mask[b]]
            nb = valid_x.size(0)
            packed_nodes.append(valid_x)
            batch_ids.append(torch.full((nb,), b, dtype=torch.long, device=device))

            if edge_indices[b].numel() > 0:
                global_edge_indices.append(edge_indices[b] + offset)
                global_edge_attrs.append(edge_attrs[b])

            offset += nb

        x_packed = torch.cat(packed_nodes, dim=0)
        batch_vec = torch.cat(batch_ids, dim=0)

        if len(global_edge_indices) > 0:
            e_idx = torch.cat(global_edge_indices, dim=1)
            e_attr = torch.cat(global_edge_attrs, dim=0)
        else:
            e_idx = torch.zeros((2, 0), dtype=torch.long, device=device)
            e_attr = torch.zeros(
                (0, self.pair_in_dim), device=device, dtype=x_packed.dtype
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
