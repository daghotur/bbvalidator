import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class BackboneGeometryExtractor(nn.Module):
    def __init__(
        self,
        clash_threshold: float = 3.5,
        hbond_threshold: float = 3.5,
        cb_length: float = 1.53,
        o_length: float = 1.23,
    ):
        super().__init__()
        self.clash_threshold = clash_threshold
        self.hbond_threshold = hbond_threshold
        self.cb_length = cb_length
        self.o_length = o_length

    @staticmethod
    def _dihedral(
        p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor
    ) -> torch.Tensor:
        b0 = -(p2 - p1)
        b1 = p3 - p2
        b2 = p4 - p3
        b1_n = b1 / (torch.linalg.norm(b1, dim=-1, keepdim=True) + 1e-8)
        v = b0 - torch.sum(b0 * b1_n, dim=-1, keepdim=True) * b1_n
        w = b2 - torch.sum(b2 * b1_n, dim=-1, keepdim=True) * b1_n
        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.linalg.cross(b1_n, v, dim=-1) * w, dim=-1)
        return torch.atan2(y, x) * (180.0 / torch.pi)

    @staticmethod
    def _pad_right(tensor: torch.Tensor, seq_len: int, pad_val: float = 0.0) -> torch.Tensor:
        """Паддинг справа до seq_len."""
        return F.pad(tensor, (0, seq_len - tensor.shape[1]), value=pad_val)

    @staticmethod
    def _pair_masks(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Маски величин, определённых на СОСЕДНЕЙ паре остатков.

        Возвращает (with_prev, with_next):
          with_prev[i] — валидны и i, и i-1 (φ, ω: паддинг слева);
          with_next[i] — валидны и i, и i+1 (ψ, ca_dist: паддинг справа).

        Считается по реальной маске сэмпла, а не по общей длине тензора:
        иначе последний валидный остаток короткой структуры в батче берёт
        координаты паддинга и получает мусорный торсион, то есть признаки
        начинают зависеть от того, с кем структура попала в батч.
        """
        pair = mask[:, 1:] & mask[:, :-1]
        with_prev = F.pad(pair, (1, 0), value=False)
        with_next = F.pad(pair, (0, 1), value=False)
        return with_prev, with_next

    def forward(
        self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, N, _, _ = coords.shape
        device = coords.device

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device)

        N_at = coords[:, :, 0, :]
        CA = coords[:, :, 1, :]
        C_at = coords[:, :, 2, :]

        # --- Торсионные углы ---
        phi_raw = self._dihedral(C_at[:, :-1], N_at[:, 1:], CA[:, 1:], C_at[:, 1:])
        psi_raw = self._dihedral(N_at[:, :-1], CA[:, :-1], C_at[:, :-1], N_at[:, 1:])
        omega_raw = self._dihedral(CA[:, :-1], C_at[:, :-1], N_at[:, 1:], CA[:, 1:])

        # Маски по реальной длине сэмпла: определённость торсиона задаёт
        # валидность соседнего остатка, а не позиция в паддированном тензоре.
        with_prev, with_next = self._pair_masks(mask)

        # phi: определён для остатков 1..L-1 → паддинг слева
        phi = F.pad(phi_raw, (1, 0), value=0.0)  # [B, N]
        phi_mask = with_prev

        # psi: определён для остатков 0..L-2 → паддинг справа
        psi = F.pad(psi_raw, (0, 1), value=0.0)  # [B, N]
        psi_mask = with_next

        # omega: определён для остатков 1..L-1 (связь i-1 → i) → паддинг слева
        omega = F.pad(omega_raw, (1, 0), value=0.0)  # [B, N]
        omega_mask = with_prev

        # --- Аутлайеры Рамачандрана ---
        # FIX: прежняя логика помечала α-спираль как аутлайер (она является допустимой областью).
        # Теперь аутлайер — остаток вне всех допустимых областей.
        in_alpha = (phi > -90.0) & (phi < -30.0) & (psi > -80.0) & (psi < -10.0)
        in_beta = (phi > -170.0) & (phi < -50.0) & (psi > 80.0) & (psi < 180.0)
        in_lalpha = (phi > 30.0) & (phi < 90.0) & (psi > 20.0) & (psi < 80.0)

        ram_outliers = ~(in_alpha | in_beta | in_lalpha) & phi_mask & psi_mask & mask

        # --- Виртуальный Cβ ---
        n_ca = N_at - CA
        c_ca = C_at - CA
        n_ca_norm = F.normalize(n_ca, dim=-1)
        c_ca_norm = F.normalize(c_ca, dim=-1)

        bisector = F.normalize(n_ca_norm + c_ca_norm, dim=-1)
        normal = F.normalize(torch.linalg.cross(n_ca_norm, c_ca_norm, dim=-1), dim=-1)

        cb_dir = -0.58 * bisector + 0.81 * normal
        virtual_cb = CA + self.cb_length * F.normalize(cb_dir, dim=-1)

        # --- Виртуальный кислород ---
        # Тригональная плоскость при C: направления на CA (u_o) и N(i+1) (v_o)
        # разведены на ~116°, поэтому карбонильный O лежит напротив их биссектрисы:
        # o_dir = -(u_o_n + v_o_n), угол O-C-CA получается ~122°.
        u_o = CA - C_at
        next_N = torch.roll(N_at, shifts=-1, dims=1)
        v_o = next_N - C_at
        # roll замыкает последний остаток на первый; на C-конце нет N(i+1) —
        # подставляем u_o (любой корректный вектор): строка O(N-1) всё равно
        # исключена из подсчёта тригональной маской (не существует j >= i+3).
        is_last = torch.zeros(B, N, dtype=torch.bool, device=device)
        is_last[:, -1] = True
        v_o = torch.where(is_last.unsqueeze(-1), u_o, v_o)
        o_dir = F.normalize(
            F.normalize(u_o, dim=-1) + F.normalize(v_o, dim=-1), dim=-1
        )
        oxygen = C_at - self.o_length * o_dir

        # --- Водородные связи (прокси по расстоянию O–N) ---
        # Схема DSSP-типа: O(i)…N(j), j >= i+3, каждая связь считается один раз.
        dist_ON = torch.cdist(oxygen, N_at)
        hbond_mask = dist_ON < self.hbond_threshold
        tri_mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=device), diagonal=3
        )
        hb_valid = (
            hbond_mask & tri_mask.unsqueeze(0) & mask.unsqueeze(2) & mask.unsqueeze(1)
        )
        hbond_count = hb_valid.sum(dim=-1, dtype=torch.float32)

        # --- Клэши по виртуальному Cβ ---
        # Пары с |i-j| < 3 — геометрия цепи, а не стерический конфликт.
        dist_cb = torch.cdist(virtual_cb, virtual_cb)
        seq_idx = torch.arange(N, device=device)
        seq_far = (seq_idx.unsqueeze(0) - seq_idx.unsqueeze(1)).abs() >= 3
        clash_mask_mat = (
            (dist_cb < self.clash_threshold)
            & seq_far.unsqueeze(0)
            & mask.unsqueeze(2)
            & mask.unsqueeze(1)
        )
        clash_count_per_residue = clash_mask_mat.sum(dim=-1, dtype=torch.float32)

        # --- Расстояния Cα–Cα ---
        ca_dist_raw = torch.linalg.norm(CA[:, 1:] - CA[:, :-1], dim=-1)
        ca_dist = self._pad_right(ca_dist_raw, N)
        ca_mask = with_next

        return {
            # Виртуальный Cβ отдаётся наружу: на нём строятся ориентационные
            # признаки рёбер (PairFeatureBuilder). В pack_for_mlp не входит.
            "virtual_cb": virtual_cb,
            "phi": phi,
            "psi": psi,
            "omega": omega,
            "phi_mask": phi_mask,
            "psi_mask": psi_mask,
            "omega_mask": omega_mask,
            "ram_outliers": ram_outliers,
            "hbond_count": hbond_count,
            "clash_count": clash_count_per_residue,
            "ca_dist": ca_dist,
            "ca_mask": ca_mask,
            "global_mask": mask,
        }

    @staticmethod
    def pack_for_mlp(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        m = feats["global_mask"].float()

        deg2rad = torch.pi / 180.0

        phi_rad = feats["phi"] * deg2rad
        psi_rad = feats["psi"] * deg2rad
        omega_rad = feats["omega"] * deg2rad

        sin_phi = torch.sin(phi_rad) * feats["phi_mask"].float()
        cos_phi = torch.cos(phi_rad) * feats["phi_mask"].float()
        sin_psi = torch.sin(psi_rad) * feats["psi_mask"].float()
        cos_psi = torch.cos(psi_rad) * feats["psi_mask"].float()
        sin_omega = torch.sin(omega_rad) * feats["omega_mask"].float()
        cos_omega = torch.cos(omega_rad) * feats["omega_mask"].float()

        ca_norm = (
            torch.clip((feats["ca_dist"] - 3.0) / 2.0, 0.0, 1.0)
            * feats["ca_mask"].float()
        )
        ram_norm = feats["ram_outliers"].float()
        hb_norm = torch.clip(feats["hbond_count"] / 4.0, 0.0, 1.0)
        clash_norm = torch.clip(feats["clash_count"] / 5.0, 0.0, 1.0)

        stacked = torch.stack(
            [
                sin_phi,
                cos_phi,
                sin_psi,
                cos_psi,
                sin_omega,
                cos_omega,
                ca_norm,
                ram_norm,
                hb_norm,
                clash_norm,
            ],
            dim=-1,
        )
        return stacked * m.unsqueeze(-1)
