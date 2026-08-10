"""
Постобработка H5-датасетов: считаем steric_target через
BackboneGeometryExtractor отдельно от параллельной сборки.

H-связи как auxiliary-таргет убраны (решение 2026-08-09): hbond_count
остаётся входной фичей фронтенда, но отдельной регрессионной головы нет.
"""

import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from preprocess import geometry_features


def _build_extractor() -> tuple[geometry_features.BackboneGeometryExtractor, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return geometry_features.BackboneGeometryExtractor().eval(), device


@torch.no_grad()
def _compute_steric(
    extractor: geometry_features.BackboneGeometryExtractor,
    coords: np.ndarray,
    device: torch.device,
) -> float:
    length = len(coords)
    coords_pt = torch.from_numpy(coords.astype(np.float32)).unsqueeze(0).to(device)
    feats = extractor(coords_pt)

    clash_total = feats["clash_count"].sum().item()
    return float(clash_total / length)


def _needs_recompute(grp: h5py.Group) -> bool:
    steric = grp.attrs.get("steric_target", None)
    if steric is None:
        return True
    return not np.isfinite(steric)


def compute_targets_for_h5(h5_path: str, *, force: bool = False) -> None:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Не найден файл: {os.path.abspath(h5_path)}")

    extractor, device = _build_extractor()

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
            grp.attrs["steric_target"] = _compute_steric(extractor, coords, device)
            updated += 1

    print(f"{h5_path}: обновлено {updated}, пропущено {skipped}")


if __name__ == "__main__":
    # Стерические таргеты нужны и позитивам, и негативам.
    for h5_name in ["positive_proteins.h5", "negative_proteins.h5"]:
        compute_targets_for_h5(h5_name)
