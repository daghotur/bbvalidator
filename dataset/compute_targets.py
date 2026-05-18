"""
Постобработка H5-датасетов: считаем steric_target и hbond_target через
BackboneGeometryExtractor отдельно от параллельной сборки.
"""
import os
from typing import Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from preprocess import geometry_features


def _build_extractor() -> geometry_features.BackboneGeometryExtractor:
    return geometry_features.BackboneGeometryExtractor().eval()


@torch.no_grad()
def _compute_for_coords(
        extractor: geometry_features.BackboneGeometryExtractor,
        coords: np.ndarray,
) -> Tuple[float, float]:
    length = len(coords)
    coords_pt = torch.from_numpy(coords.astype(np.float32)).unsqueeze(0)
    feats = extractor(coords_pt)

    clash_total = feats["clash_count"].sum().item()
    steric_val = clash_total / length

    hbonds_total = feats["hbond_count"].sum().item()
    hbond_val = max(0.0, (length - hbonds_total) / length)

    return float(steric_val), float(hbond_val)


def _needs_recompute(grp: h5py.Group) -> bool:
    steric = grp.attrs.get("steric_target", None)
    hbond = grp.attrs.get("hbond_target", None)
    if steric is None or hbond is None:
        return True
    return not (np.isfinite(steric) and np.isfinite(hbond))


def compute_targets_for_h5(h5_path: str, *, force: bool = False) -> None:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Не найден файл: {os.path.abspath(h5_path)}")

    extractor = _build_extractor()

    with h5py.File(h5_path, "r+") as h5f:
        keys = list(h5f.keys())
        updated = 0
        skipped = 0

        for key in tqdm(keys, desc=os.path.basename(h5_path)):
            grp = h5f[key]
            if "coords" not in grp:
                continue

            if not force and not _needs_recompute(grp):
                skipped += 1
                continue

            coords = grp["coords"][:]
            steric_val, hbond_val = _compute_for_coords(extractor, coords)
            grp.attrs["steric_target"] = steric_val
            grp.attrs["hbond_target"] = hbond_val
            updated += 1

    print(f"{h5_path}: обновлено {updated}, пропущено {skipped}")


if __name__ == "__main__":
    compute_targets_for_h5("negative_proteins.h5")
