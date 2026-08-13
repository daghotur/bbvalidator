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

    # Ориентационные признаки на ребро (набор trRosetta), см. _orientation:
    # sin/cos omega, sin/cos theta_ij, sin/cos theta_ji, cos phi_ij, cos phi_ji,
    # нормированная разность d(Cb) - d(Ca).
    ORIENT_DIM = 9

    @property
    def feature_dim(self) -> int:
        # rbf(bins) + contact + seq_sep_norm + seq_adjacent + same_local_window
        return self.rbf.bins + 4 + self.ORIENT_DIM

    @staticmethod
    def _dihedral(p1: torch.Tensor, p2: torch.Tensor,
                  p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """Двугранный угол в радианах для наборов точек формы [E, 3]."""
        b0 = -(p2 - p1)
        b1 = p3 - p2
        b2 = p4 - p3
        b1n = b1 / (torch.linalg.norm(b1, dim=-1, keepdim=True) + 1e-8)
        v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
        w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
        return torch.atan2(
            (torch.linalg.cross(b1n, v, dim=-1) * w).sum(-1), (v * w).sum(-1)
        )

    @staticmethod
    def _angle(a: torch.Tensor, vertex: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Косинус планарного угла a-vertex-c."""
        u = a - vertex
        v = c - vertex
        u = u / (torch.linalg.norm(u, dim=-1, keepdim=True) + 1e-8)
        v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-8)
        return (u * v).sum(-1)

    def _orientation(
        self,
        ca_i: torch.Tensor, cb_i: torch.Tensor, n_i: torch.Tensor,
        ca_j: torch.Tensor, cb_j: torch.Tensor, n_j: torch.Tensor,
        dist_ca: torch.Tensor,
    ) -> torch.Tensor:
        """Взаимная ориентация остатков пары — набор trRosetta. [E, ORIENT_DIM]

        Зачем: матрица расстояний Cα инвариантна к отражению, поэтому парный
        канал без углов ахирален. Правизна β-α-β перекрёстов, закрутка β-листа
        и углы упаковки спиралей в нём не представлены (измерено: ранжирование
        переживает зеркалирование структуры на Spearman 0.41). Двугранные
        omega и theta меняют знак при отражении — это и чинит хиральность.

        omega    — двугранный CA_i-CB_i-CB_j-CA_j, симметричен по i, j;
        theta_ij — двугранный N_i-CA_i-CB_i-CB_j, асимметричен;
        phi_ij   — планарный угол CA_i-CB_i-CB_j (даём косинусом: он на [0, pi],
                   монотонен, второй компонент был бы избыточен).
        """
        omega = self._dihedral(ca_i, cb_i, cb_j, ca_j)
        theta_ij = self._dihedral(n_i, ca_i, cb_i, cb_j)
        theta_ji = self._dihedral(n_j, ca_j, cb_j, cb_i)
        d_cb = torch.linalg.norm(cb_i - cb_j, dim=-1)
        return torch.stack(
            [
                torch.sin(omega), torch.cos(omega),
                torch.sin(theta_ij), torch.cos(theta_ij),
                torch.sin(theta_ji), torch.cos(theta_ji),
                self._angle(ca_i, cb_i, cb_j),
                self._angle(ca_j, cb_j, cb_i),
                # разность d(Cb) - d(Ca): расходятся боковые цепи или сходятся
                torch.clamp((d_cb - dist_ca) / 2.0, -1.0, 1.0),
            ],
            dim=-1,
        )

    def _edge_attr(
        self,
        dist_e: torch.Tensor,  # [E] — расстояния Cα–Cα для рёбер
        src_global: torch.Tensor,  # [E] — глобальные seq-индексы источника
        dst_global: torch.Tensor,  # [E] — глобальные seq-индексы приёмника
        orient: torch.Tensor,  # [E, ORIENT_DIM] — взаимная ориентация
    ) -> torch.Tensor:
        rbf_e = self.rbf(dist_e)  # [E, bins]
        contact = (dist_e < 8.0).float().unsqueeze(-1)

        seq_sep = torch.abs(src_global - dst_global).float()
        seq_sep_norm = torch.clamp(seq_sep / 32.0, 0.0, 1.0).unsqueeze(-1)
        seq_adjacent = (seq_sep == 1).float().unsqueeze(-1)
        same_local_window = (seq_sep <= 4).float().unsqueeze(-1)

        return torch.cat(
            [rbf_e, contact, seq_sep_norm, seq_adjacent, same_local_window, orient],
            dim=-1,
        )  # [E, F_pair]

    def forward(
        self,
        ca_coords: torch.Tensor,
        mask: torch.Tensor,
        dist_mat: Optional[torch.Tensor] = None,
        cb_coords: Optional[torch.Tensor] = None,
        n_coords: Optional[torch.Tensor] = None,
    ) -> Dict:
        """kNN-граф всего батча одним куском, без цикла по образцам.

        Рёбра сразу выдаются в «упакованной» нумерации: валидные узлы всех
        образцов сложены подряд в row-major порядке, как их достаёт x[mask].
        Прежняя версия строила список тензоров на образец и склеивала его в
        энкодере; цикл по батчу стоил 95% времени этого компонента и около
        трети всего инференса при B=32 (замер 2026-08-13).
        """
        B, N, _ = ca_coords.shape
        device = ca_coords.device

        # 1. Попарные расстояния (нужны только для построения kNN-графа)
        if dist_mat is None:
            dist_mat = torch.cdist(ca_coords, ca_coords)

        # 2. kNN-граф. Признаки рёбер считаются напрямую для найденных рёбер,
        #    без материализации плотного тензора [B, N, N, F_pair] (экономия O(N²)).
        k = min(self.k_neighbors, N - 1)

        pair_valid = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        dist_knn = dist_mat.masked_fill(~pair_valid | eye, float("inf"))

        knn_vals, knn_idx = dist_knn.topk(k=k, largest=False)  # [B, N, k]

        # 3. Позиция каждого валидного узла в упакованном представлении.
        #    Паддинг получает -1 и в рёбра не попадает: источники отсекаются
        #    маской, приёмники — тем, что расстояние до паддинга равно inf.
        packed_pos = torch.full((B, N), -1, dtype=torch.long, device=device)
        n_valid = int(mask.sum())
        packed_pos[mask] = torch.arange(n_valid, device=device)

        # 4. Живые рёбра: источник валиден и сосед — не паддинг
        is_real = knn_vals.isfinite() & mask.unsqueeze(-1)  # [B, N, k]

        src_packed = packed_pos.unsqueeze(-1).expand(B, N, k)[is_real]
        dst_packed = torch.gather(
            packed_pos, 1, knn_idx.reshape(B, N * k)
        ).reshape(B, N, k)[is_real]

        seq_i = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)

        # 5. Ориентация — только на найденных рёбрах ([E] штук вместо плотных
        #    [B, N, N]), поэтому стоит в пределах шума на фоне kNN.
        batch_e = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)[is_real]
        i_e = seq_i[is_real]
        j_e = knn_idx[is_real]
        if cb_coords is None or n_coords is None:
            orient = torch.zeros(
                (i_e.numel(), self.ORIENT_DIM), device=device, dtype=ca_coords.dtype
            )
        else:
            orient = self._orientation(
                ca_coords[batch_e, i_e], cb_coords[batch_e, i_e], n_coords[batch_e, i_e],
                ca_coords[batch_e, j_e], cb_coords[batch_e, j_e], n_coords[batch_e, j_e],
                knn_vals[is_real],
            )

        edge_attr = self._edge_attr(knn_vals[is_real], i_e, j_e, orient)

        return {
            "edge_index": torch.stack([src_packed, dst_packed], dim=0),  # [2, E]
            "edge_attr": edge_attr,  # [E, F_pair]
            "n_valid": n_valid,
        }
