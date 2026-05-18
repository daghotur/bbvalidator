import os
import threading
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

# Per-worker HDF5 file handle cache (thread-safe, не требует повторного открытия)
_h5_cache = threading.local()


def _open_h5(path: str) -> h5py.File:
    if not hasattr(_h5_cache, "files"):
        _h5_cache.files = {}
    if path not in _h5_cache.files:
        _h5_cache.files[path] = h5py.File(path, "r", swmr=True)
    return _h5_cache.files[path]


class ProteinManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        augment_se3: bool = False,
        center_coords: bool = True,
        return_metadata: bool = True,
    ) -> None:
        super().__init__()

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {os.path.abspath(manifest_path)}"
            )

        df = pd.read_csv(manifest_path)
        if "split" not in df.columns:
            raise ValueError("Manifest must contain column 'split'")

        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.augment_se3 = augment_se3
        self.center_coords = center_coords
        self.return_metadata = return_metadata

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split='{split}' in {manifest_path}")

        print(
            f"Dataset '{split}' loaded: {len(self.df)} samples | "
            f"augment_se3={self.augment_se3} | center_coords={self.center_coords}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"ProteinManifestDataset(split={self.split!r}, n={len(self.df)}, "
            f"augment_se3={self.augment_se3})"
        )

    @staticmethod
    def _get(row, key: str, default=None):
        if key not in row or pd.isna(row[key]):
            return default
        return row[key]

    @staticmethod
    def _center_coords(coords: np.ndarray) -> np.ndarray:
        centroid = coords.mean(axis=(0, 1), keepdims=True)  # [1, 1, 3]
        return coords - centroid

    @staticmethod
    def _apply_random_rotation(coords: np.ndarray) -> np.ndarray:
        rot = Rotation.random().as_matrix().astype(np.float32)  # [3, 3]
        rotated = coords.reshape(-1, 3) @ rot.T
        return rotated.reshape(coords.shape).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:  # ty:ignore[invalid-method-override]
        row = self.df.iloc[idx]

        h5_path = str(row["source_h5"])
        group_key = str(row["h5_group_key"])
        label = float(row["label"])

        # Переиспользуем открытый дескриптор файла внутри одного worker-процесса
        h5f = _open_h5(h5_path)
        if group_key not in h5f:
            raise KeyError(f"Group '{group_key}' not found in {h5_path}")
        grp = h5f[group_key]
        if "coords" not in grp:
            raise KeyError(
                f"Dataset 'coords' missing in group '{group_key}' ({h5_path})"
            )

        coords = grp["coords"][:].astype(np.float32)  # [L, 3, 3]

        if self.center_coords:
            coords = self._center_coords(coords)
        if self.augment_se3:
            coords = self._apply_random_rotation(coords)

        rmsd_target = float(grp.attrs.get("rmsd_target", 0.0))
        steric_target = float(grp.attrs.get("steric_target", 0.0))
        hbond_target = float(grp.attrs.get("hbond_target", 0.0))
        failure_mode = int(grp.attrs.get("failure_mode_label", 0))

        item: Dict = {
            "coords": torch.from_numpy(coords),
            "label": torch.tensor([label], dtype=torch.float32),
            "rmsd_target": torch.tensor(rmsd_target, dtype=torch.float32),
            "steric_target": torch.tensor(steric_target, dtype=torch.float32),
            "hbond_target": torch.tensor(hbond_target, dtype=torch.float32),
            "failure_mode_label": torch.tensor(failure_mode, dtype=torch.long),
            "length": int(coords.shape[0]),
        }

        if self.return_metadata:
            item["sample_key"] = str(self._get(row, "sample_key", f"sample::{idx}"))
            item["group_id"] = str(self._get(row, "group_id", group_key))
            item["parent_positive_key"] = str(
                self._get(row, "parent_positive_key", group_key)
            )
            item["strategy"] = str(self._get(row, "strategy", "unknown"))
            item["source_h5"] = h5_path
            item["h5_group_key"] = group_key
            item["pdb_id"] = self._get(row, "pdb_id", None)
            item["chain_id"] = self._get(row, "chain_id", None)
            item["method"] = self._get(row, "method", None)

        return item


def protein_collate_fn(batch: List[Dict]) -> Dict:
    """Динамический collate для белков разной длины (паддинг до max_len батча)."""
    max_len = max(item["length"] for item in batch)
    B = len(batch)

    coords_padded = torch.zeros((B, max_len, 3, 3), dtype=torch.float32)
    mask = torch.zeros((B, max_len), dtype=torch.bool)

    labels = torch.zeros((B, 1), dtype=torch.float32)
    lengths = torch.zeros((B,), dtype=torch.long)
    rmsd_targets = torch.zeros((B,), dtype=torch.float32)
    steric_targets = torch.zeros((B,), dtype=torch.float32)
    hbond_targets = torch.zeros((B,), dtype=torch.float32)
    failure_labels = torch.zeros((B,), dtype=torch.long)

    meta_keys = [
        "sample_key",
        "group_id",
        "parent_positive_key",
        "strategy",
        "pdb_id",
        "chain_id",
        "method",
        "source_h5",
        "h5_group_key",
    ]
    meta: Dict[str, list] = {k: [] for k in meta_keys}

    for i, item in enumerate(batch):
        L = int(item["length"])
        coords_padded[i, :L] = item["coords"]
        mask[i, :L] = True

        rmsd_targets[i] = item["rmsd_target"]
        steric_targets[i] = item["steric_target"]
        hbond_targets[i] = item["hbond_target"]
        failure_labels[i] = item["failure_mode_label"]
        labels[i] = item["label"]
        lengths[i] = L
        for k in meta_keys:
            meta[k].append(item.get(k, None))

    return {
        "coords": coords_padded,
        "mask": mask,
        "label": labels,
        "rmsd_target": rmsd_targets,
        "steric_target": steric_targets,
        "hbond_target": hbond_targets,
        "failure_mode_label": failure_labels,
        "length": lengths,
        **meta,
    }


def make_loader(
    manifest_path: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    augment_se3: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    dataset = ProteinManifestDataset(
        manifest_path=manifest_path,
        split=split,
        augment_se3=augment_se3,
        center_coords=True,
        return_metadata=True,
    )

    if shuffle is None:
        shuffle = split == "train"
    use_persistent = persistent_workers and num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=protein_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        drop_last=drop_last,
    )


def get_dataloaders(
    manifest_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = make_loader(
        manifest_path,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        augment_se3=True,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        manifest_path,
        split="val",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        augment_se3=False,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        manifest_path,
        split="test",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        augment_se3=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    MANIFEST_PATH = "manifest_v1_split.csv"

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            MANIFEST_PATH,
            batch_size=128,
            num_workers=8,
            pin_memory=False,
        )
        batch = next(iter(train_loader))

        print("\n=== DATALOADER CHECK ===")
        print(f"coords shape : {batch['coords'].shape}  -> [B, Nmax, 3, 3]")
        print(f"mask shape   : {batch['mask'].shape}    -> [B, Nmax]")
        print(f"length shape : {batch['length'].shape}  -> [B]")
        print(f"labels       : {batch['label'].view(-1).tolist()}")
        print(f"lengths      : {batch['length'].tolist()}")
        print(f"rmsd_target   : {batch['rmsd_target'].tolist()}")
        print(f"steric_target : {batch['steric_target'].tolist()}")
        print(f"failure_modes : {batch['failure_mode_label'].tolist()}")
        print(f"strategies   : {batch['strategy']}")

    except FileNotFoundError:
        print(f"Manifest file '{MANIFEST_PATH}' not found.")
