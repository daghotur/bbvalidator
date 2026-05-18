import torch
import torch.nn as nn
import contextlib
from typing import Dict

from .geometry_features import BackboneGeometryExtractor
from .foldability_features import FoldabilityProxies
from .pair_features import PairFeatureBuilder


class BiophysicalFrontend(nn.Module):
    def __init__(self, use_no_grad: bool = True):
        super().__init__()
        self.geometry = BackboneGeometryExtractor()
        self.foldability = FoldabilityProxies(pca_components=16)
        self.pair_builder = PairFeatureBuilder(rbf_bins=16, k_neighbors=16)
        self.use_no_grad = use_no_grad

    def forward(
            self, coords: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        ctx = torch.no_grad() if self.use_no_grad else contextlib.nullcontext()

        with ctx:
            ca = coords[:, :, 1, :]  # [B, N, 3]
            dist_mat = torch.cdist(ca, ca)  # [B, N, N]

            # 1. Сырые признаки
            geom_raw = self.geometry(coords, mask)
            fold_raw = self.foldability(ca, mask, dist_mat=dist_mat)

            # 2. Пары и граф
            # 2. Пары и граф
            pair_data = self.pair_builder(
                ca, mask, dist_mat=dist_mat
            )

            # 3. Узловые признаки
            geom_x = self.geometry.pack_for_mlp(geom_raw)  # [B, N, 10]
            fold_x = self.foldability.pack_for_mlp(fold_raw)  # [B, N, 21]

            node_feats = torch.cat([geom_x, fold_x], dim=-1)  # [B, N, 31]

            return {
                "node_feats": node_feats,  # [B, N, F_node]
                "pair_feats": pair_data["pair_feats"],  # [B, N, N, F_pair]
                "edge_indices": pair_data["edge_indices"],  # List[Tensor(2, E)]
                "edge_attrs": pair_data["edge_attrs"],  # List[Tensor(E, F_pair)]
                "mask": mask,
            }
