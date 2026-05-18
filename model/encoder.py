"""
model/encoder.py
----------------
HybridProteinEncoder: Graph message-passing (MPNN) → Pre-LayerNorm Transformer.

Входной контракт (dict от BiophysicalFrontend):
  node_features  : [B, N, node_in_dim]
  pair_features  : [B, N, N, pair_in_dim]
  mask           : [B, N]  bool — True = валидный остаток, False = паддинг

Выход: node embeddings [B, N, d_model]
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Graph layer
# ---------------------------------------------------------------------------

class PyGGraphMessageLayer(MessagePassing):
    def __init__(self, d_model: int, pair_dim: int, dropout: float = 0.1):
        super().__init__(aggr='mean')

        self.node_proj = nn.Linear(d_model, d_model, bias=False)
        self.pair_proj = nn.Linear(pair_dim, d_model)
        self.msg_act = nn.GELU()
        self.msg_out = nn.Linear(d_model, d_model)

        self.gate_z = nn.Linear(d_model * 2, d_model)
        self.gate_r = nn.Linear(d_model * 2, d_model)

        # ИСПРАВЛЕНО: Избегаем конфликта имен с встроенным MessagePassing.update
        self.node_update = nn.Linear(d_model * 2, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,  # Упакованные узлы [Total_valid_nodes, d_model]
            edge_index: torch.Tensor,  # [2, Total_edges]
            edge_attr: torch.Tensor  # [Total_edges, pair_dim]
    ) -> torch.Tensor:
        # propagate вызывает self.message(), а затем усредняет (aggr='mean')
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        agg = self.dropout(agg)

        # GRU-подобное обновление
        xz = torch.cat([x, agg], dim=-1)
        z = torch.sigmoid(self.gate_z(xz))
        r = torch.sigmoid(self.gate_r(xz))

        # ИСПРАВЛЕНО: Вызываем переименованный слой
        h = torch.tanh(self.node_update(torch.cat([r * x, agg], dim=-1)))

        out = (1.0 - z) * x + z * h
        return self.norm(out)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h_i = self.node_proj(x_i)
        h_j = self.node_proj(x_j)
        h_e = self.pair_proj(edge_attr)

        msg = self.msg_act(h_i + h_j + h_e)
        return self.msg_out(msg)


# ---------------------------------------------------------------------------
# Hybrid encoder
# ---------------------------------------------------------------------------

class HybridProteinEncoder(nn.Module):
    def __init__(
            self,
            node_in_dim: int,
            pair_in_dim: int,
            d_model: int,
            pair_dim: int,
            num_graph_layers: int,
            num_transformer_layers: int,
            num_heads: int,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.pair_embed = nn.Sequential(
            nn.Linear(pair_in_dim, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
        )

        # PyG слои
        self.graph_layers = nn.ModuleList([
            PyGGraphMessageLayer(d_model, pair_dim, dropout)
            for _ in range(num_graph_layers)
        ])

        # Трансформер
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,
        )

    def forward(self, features: dict) -> torch.Tensor:
        x_raw = features["node_feats"]  # [B, N, node_in_dim]
        mask = features["mask"]  # [B, N] bool
        edge_indices = features["edge_indices"]  # List[Tensor(2, E_b)]
        edge_attrs = features["edge_attrs"]  # List[Tensor(E_b, pair_in_dim)]

        B, N, _ = x_raw.shape
        device = x_raw.device

        # 1. Проекция узлов
        x = self.node_embed(x_raw)  # [B, N, d_model]

        # 2. Упаковка графа (убираем паддинг для MPNN)
        packed_nodes = []
        global_edge_indices = []
        global_edge_attrs = []
        offset = 0

        for b in range(B):
            # Берем только валидные узлы из текущего графа в батче
            valid_x = x[b][mask[b]]  # [nb, d_model]
            nb = valid_x.size(0)
            packed_nodes.append(valid_x)

            if edge_indices[b].numel() > 0:
                # Смещаем индексы графа на текущий offset (так как сплющиваем все в 1 массив)
                global_edge_indices.append(edge_indices[b] + offset)
                global_edge_attrs.append(edge_attrs[b])

            offset += nb

        x_packed = torch.cat(packed_nodes, dim=0)  # [Total_valid_nodes, d_model]

        # 3. Графовые слои
        if len(global_edge_indices) > 0:
            e_idx = torch.cat(global_edge_indices, dim=1)  # [2, Total_edges]
            e_attr = torch.cat(global_edge_attrs, dim=0)  # [Total_edges, pair_in_dim]

            # Проецируем пары только один раз (для валидных ребер)
            e_attr = self.pair_embed(e_attr)

            for layer in self.graph_layers:
                # Используем gradient checkpointing для жесткой экономии памяти
                x_packed = checkpoint(layer, x_packed, e_idx, e_attr, use_reentrant=False)

        # 4. Распаковка обратно в форму батча для Трансформера
        x_out = torch.zeros((B, N, x.size(-1)), device=device, dtype=x.dtype)
        x_out[mask] = x_packed  # Раскидываем вычисленные признаки на валидные позиции

        # 5. Трансформер (FlashAttention активируется автоматически с bfloat16)
        pad_mask = ~mask  # True = padding
        x_out = self.transformer(x_out, src_key_padding_mask=pad_mask)

        return x_out
