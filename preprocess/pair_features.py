import torch
import torch.nn as nn
from typing import Dict, List, Optional


class RBFExpansion(nn.Module):
    def __init__(self, vmin: float = 2.0, vmax: float = 22.0, bins: int = 16):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        centers = torch.linspace(vmin, vmax, bins)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / ((vmax - vmin) / bins) ** 2

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * (diff**2))


class PairFeatureBuilder(nn.Module):
    def __init__(self, rbf_bins: int = 16, k_neighbors: int = 16):
        super().__init__()
        self.rbf = RBFExpansion(bins=rbf_bins)
        self.k_neighbors = k_neighbors

    def forward(
        self,
        ca_coords: torch.Tensor,
        mask: torch.Tensor,
        dist_mat: Optional[
            torch.Tensor
        ] = None
    ) -> Dict:
        B, N, _ = ca_coords.shape
        device = ca_coords.device

        # 1. Попарные расстояния
        if dist_mat is None:
            dist_mat = torch.cdist(ca_coords, ca_coords)

        rbf_feats = self.rbf(dist_mat)  # [B, N, N, bins]

        # 2. Признаки пар
        contact_flag = (dist_mat < 8.0).float().unsqueeze(-1)

        seq_idx = torch.arange(N, device=device)
        seq_sep_raw = torch.abs(seq_idx.unsqueeze(0) - seq_idx.unsqueeze(1))
        seq_sep_raw = seq_sep_raw.unsqueeze(0).expand(B, -1, -1)

        seq_sep_norm = torch.clamp(seq_sep_raw.float() / 32.0, 0.0, 1.0).unsqueeze(-1)
        seq_adjacent = (seq_sep_raw == 1).float().unsqueeze(-1)
        same_local_window = (seq_sep_raw <= 4).float().unsqueeze(-1)

        pair_feats = torch.cat(
            [rbf_feats, contact_flag, seq_sep_norm, seq_adjacent, same_local_window],
            dim=-1,
        )  # [B, N, N, F_pair]

        # 3. kNN-граф
        k = min(self.k_neighbors, N - 1)

        pair_valid = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        dist_knn = dist_mat.masked_fill(~pair_valid | eye, float("inf"))

        knn_vals, knn_idx = dist_knn.topk(k=k, largest=False)  # [B, N, k]

        edge_indices: List[torch.Tensor] = []
        edge_attrs: List[torch.Tensor] = []

        for b in range(B):
            valid = mask[b]  # [N]
            nb = int(valid.sum())
            if nb < 2:
                edge_indices.append(
                    torch.zeros((2, 0), dtype=torch.long, device=device)
                )
                edge_attrs.append(torch.zeros((0, pair_feats.shape[-1]), device=device))
                continue

            # Глобальные позиции валидных узлов и отображение global → local
            valid_pos = valid.nonzero(as_tuple=False).view(-1)  # [nb]
            g2l = torch.full((N,), -1, dtype=torch.long, device=device)
            g2l[valid_pos] = torch.arange(nb, device=device)

            # Для валидных узлов берём их k ближайших соседей
            knn_b_vals = knn_vals[b][valid]  # [nb, k]
            knn_b_idx = knn_idx[b][valid]  # [nb, k] — глобальные индексы dst

            # Отфильтровываем рёбра к паддингу (расстояние = inf)
            is_real = knn_b_vals.isfinite()  # [nb, k]

            src_local = (
                torch.arange(nb, device=device).unsqueeze(1).expand(nb, k)[is_real]
            )
            dst_local = g2l[knn_b_idx[is_real]]

            edge_index = torch.stack([src_local, dst_local], dim=0)
            # Признаки ребра: из плотной матрицы [nb, nb]
            e_attr = pair_feats[b][valid][:, valid][src_local, dst_local]

            edge_indices.append(edge_index)
            edge_attrs.append(e_attr)

        return {
            "pair_feats": pair_feats,  # [B, N, N, F_pair] — для трансформера
            "edge_indices": edge_indices,  # List[Tensor(2, E)] — для графовых слоёв
            "edge_attrs": edge_attrs,  # List[Tensor(E, F_pair)]
        }
