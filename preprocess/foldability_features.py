import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FoldabilityProxies(nn.Module):
    def __init__(
        self,
        contact_threshold: float = 8.0,
        seq_sep: int = 3,
        pca_components: int = 16,
        fragment_size: int = 9,
    ):
        super().__init__()
        self.contact_threshold = contact_threshold
        self.seq_sep = seq_sep
        self.fragment_size = fragment_size

        self.frag_pairs = (fragment_size * (fragment_size - 1)) // 2

        triu = torch.triu_indices(fragment_size, fragment_size, offset=1)
        self.register_buffer("triu_indices", triu)

        # PCA-проекция попарных расстояний 9-остаточных фрагментов.
        # Веса и среднее инициализируются скриптом preprocess/fit_pca.py
        # по нативным структурам, после чего проекция замораживается через
        # freeze_pca(). До fit-а веса случайные, среднее нулевое.
        self.pca_proj = nn.Linear(self.frag_pairs, pca_components, bias=False)
        self.register_buffer("frag_mean", torch.zeros(self.frag_pairs))

    def freeze_pca(self) -> None:
        """Заморозить PCA-проекцию: веса не будут обновляться оптимизатором."""
        self.pca_proj.weight.requires_grad_(False)

    def unfreeze_pca(self) -> None:
        self.pca_proj.weight.requires_grad_(True)

    def _get_local_pairwise_distances(self, coords: torch.Tensor) -> torch.Tensor:
        B, N, _ = coords.shape
        K = self.fragment_size

        pad_left = K // 2
        pad_right = K - 1 - pad_left

        coords_padded = F.pad(
            coords.transpose(1, 2), (pad_left, pad_right), mode="replicate"
        ).transpose(1, 2)
        windows = coords_padded.unfold(dimension=1, size=K, step=1)  # [B, N, 3, K]
        windows = windows.permute(0, 1, 3, 2).contiguous()  # [B, N, K, 3]

        dist_mat = torch.cdist(windows, windows)  # [B, N, K, K]

        flat_distances = dist_mat[
            :, :, self.triu_indices[0], self.triu_indices[1]
        ]  # [B, N, frag_pairs]

        return flat_distances

    def forward(
        self,
        ca_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dist_mat: Optional[
            torch.Tensor
        ] = None,  # FIX: принимаем снаружи, чтобы не вычислять дважды
    ) -> Dict[str, torch.Tensor]:
        B, N, _ = ca_coords.shape
        device = ca_coords.device

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device)

        # 1. Contact Density
        if dist_mat is None:
            dist_mat = torch.cdist(ca_coords, ca_coords)

        idx = torch.arange(N, device=device)
        seq_dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        valid_pairs = (seq_dist > self.seq_sep).unsqueeze(0).expand(B, -1, -1)
        valid_pairs = valid_pairs & mask.unsqueeze(1) & mask.unsqueeze(2)

        contacts_r6 = (dist_mat < 6.0) & valid_pairs
        contacts_r8 = (dist_mat < 8.0) & valid_pairs
        contacts_r10 = (dist_mat < 10.0) & valid_pairs

        packing_r6 = contacts_r6.sum(dim=-1).float()
        packing_r8 = contacts_r8.sum(dim=-1).float()
        packing_r10 = contacts_r10.sum(dim=-1).float()

        # 2. Surface Exposure Proxy
        mask_float = mask.float().unsqueeze(-1)
        centroid = (ca_coords * mask_float).sum(dim=1, keepdim=True) / (
            mask_float.sum(dim=1, keepdim=True) + 1e-8
        )
        dist_to_centroid = torch.linalg.vector_norm(ca_coords - centroid, dim=-1)

        N_eff = mask.sum(dim=1, keepdim=True).float()
        expected_radius = 2.2 * (N_eff**0.333) + 1e-5
        relative_burial = dist_to_centroid / expected_radius

        # 3. PDB-Fragment Similarity
        fragment_features = self._get_local_pairwise_distances(
            ca_coords
        )  # [B, N, frag_pairs]
        pca_projection = self.pca_proj(fragment_features - self.frag_mean)

        # 4. Loop Geometry Flags (stride-4 displacement — прокси кривизны)
        delta = ca_coords[:, 4:] - ca_coords[:, :-4]  # [B, N-4, 3]
        local_bending = torch.linalg.vector_norm(delta, dim=-1)
        local_bending = F.pad(local_bending, (2, 2), value=0.0)  # [B, N]

        m = mask.float()
        return {
            "packing_r6": packing_r6 * m,
            "packing_r8": packing_r8 * m,
            "packing_r10": packing_r10 * m,
            "relative_burial": relative_burial * m,
            "local_bending": local_bending * m,
            "pca_projection": pca_projection * m.unsqueeze(-1),
            "global_mask": mask,
        }

    @staticmethod
    def pack_for_mlp(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        m = feats["global_mask"].float()

        p6_norm = torch.clamp(feats["packing_r6"] / 10.0, 0.0, 1.0)
        p8_norm = torch.clamp(feats["packing_r8"] / 25.0, 0.0, 1.0)
        p10_norm = torch.clamp(feats["packing_r10"] / 50.0, 0.0, 1.0)
        burial_norm = torch.clamp(feats["relative_burial"], 0.0, 2.0)
        bend_norm = torch.clamp(feats["local_bending"] / 14.0, 0.0, 1.0)

        pca_norm = F.layer_norm(
            feats["pca_projection"],
            normalized_shape=[feats["pca_projection"].shape[-1]],
        )

        scalar_feats = torch.stack(
            [p6_norm, p8_norm, p10_norm, burial_norm, bend_norm], dim=-1
        )
        return torch.cat([scalar_feats, pca_norm], dim=-1) * m.unsqueeze(-1)
