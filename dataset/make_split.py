import json
import os

import h5py
import numpy as np
import pandas as pd

POS_H5 = "positive_proteins.h5"
NEG_H5 = "negative_proteins.h5"

OUT_MANIFEST = "manifest_v1.csv"
OUT_MANIFEST_SPLIT = "manifest_v1_split.csv"
OUT_STATS = "split_stats_v1.json"

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC вычисляется как остаток: 1 - TRAIN_FRAC - VAL_FRAC


def _safe_attr(attrs, key, default=None):
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def read_positive_manifest(pos_h5_path: str) -> list[dict]:
    rows = []
    with h5py.File(pos_h5_path, "r") as h5f:
        for key in h5f.keys():
            grp = h5f[key]
            if "coords" not in grp:
                continue
            coords_shape = tuple(grp["coords"].shape)
            rows.append(
                {
                    "sample_key": f"pos::{key}",
                    "source_h5": os.path.abspath(pos_h5_path),
                    "h5_group_key": key,
                    "label": 1,
                    "group_id": key,
                    "parent_positive_key": key,
                    "strategy": "positive_real",
                    "length": int(grp.attrs.get("length", coords_shape[0])),
                    "coords_shape": str(coords_shape),
                    "pdb_id": grp.attrs.get("pdb_id", None),
                    "chain_id": grp.attrs.get("chain_id", None),
                    "method": grp.attrs.get("method", None),
                    "decoy_index": None,
                    "random_seed": None,
                    "rmsd_target": float(_safe_attr(grp.attrs, "rmsd_target", 0.0)),
                    "steric_target": float(_safe_attr(grp.attrs, "steric_target", 0.0)),
                    "hbond_target": float(_safe_attr(grp.attrs, "hbond_target", 0.0)),
                    "failure_mode_label": int(
                        _safe_attr(grp.attrs, "failure_mode_label", 0)
                    ),
                }
            )
    return rows


def read_negative_manifest(neg_h5_path: str) -> list[dict]:
    rows = []
    with h5py.File(neg_h5_path, "r") as h5f:
        for key in h5f.keys():
            grp = h5f[key]
            if "coords" not in grp or "label" not in grp:
                continue

            parent_key = _safe_attr(grp.attrs, "source_positive_key", None)
            if parent_key is None:
                continue

            coords_shape = tuple(grp["coords"].shape)
            rows.append(
                {
                    "sample_key": f"neg::{key}",
                    "source_h5": os.path.abspath(neg_h5_path),
                    "h5_group_key": key,
                    "label": 0,
                    "group_id": parent_key,
                    "parent_positive_key": parent_key,
                    "strategy": _safe_attr(grp.attrs, "strategy", "unknown_negative"),
                    "length": int(_safe_attr(grp.attrs, "length", coords_shape[0])),
                    "coords_shape": str(coords_shape),
                    "pdb_id": None,
                    "chain_id": None,
                    "method": None,
                    "decoy_index": _safe_attr(grp.attrs, "decoy_index", None),
                    "random_seed": _safe_attr(grp.attrs, "random_seed", None),
                    "rmsd_target": float(_safe_attr(grp.attrs, "rmsd_target", 0.0)),
                    "steric_target": float(_safe_attr(grp.attrs, "steric_target", 0.0)),
                    "hbond_target": float(_safe_attr(grp.attrs, "hbond_target", 0.0)),
                    "failure_mode_label": int(
                        _safe_attr(grp.attrs, "failure_mode_label", 5)
                    ),
                }
            )
    return rows


def build_manifest(pos_h5_path: str, neg_h5_path: str) -> pd.DataFrame:
    rows = read_positive_manifest(pos_h5_path) + read_negative_manifest(neg_h5_path)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Manifest is empty: no valid samples found in input H5 files.")
    return df


def assign_group_splits(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    groups = (
        df.groupby("group_id", as_index=False)
        .agg(
            n_samples=("sample_key", "count"),
            n_pos=("label", lambda x: int((x == 1).sum())),
            n_neg=("label", lambda x: int((x == 0).sum())),
            mean_length=("length", "mean"),
        )["group_id"]
        .tolist()
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    n_test = n - n_train - n_val

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    # Явные проверки вместо assert (устойчивы к запуску с -O)
    if len(train_groups | val_groups | test_groups) != n:
        raise ValueError("Сумма групп по сплитам не совпадает с общим числом групп")
    if train_groups & val_groups:
        raise ValueError("Пересечение train/val групп обнаружено!")
    if train_groups & test_groups:
        raise ValueError("Пересечение train/test групп обнаружено!")
    if val_groups & test_groups:
        raise ValueError("Пересечение val/test групп обнаружено!")
    if n_test < 1:
        raise ValueError(
            f"Тестовый сплит пустой (n_groups={n}, train={n_train}, val={n_val})"
        )

    split_map = {g: "train" for g in train_groups}
    split_map.update({g: "val" for g in val_groups})
    split_map.update({g: "test" for g in test_groups})

    df = df.copy()
    df["split"] = df["group_id"].map(split_map)
    return df


def validate_split(df: pd.DataFrame) -> None:
    max_splits = df.groupby("group_id")["split"].nunique().max()
    if max_splits != 1:
        raise ValueError(
            f"Data leakage: group_id встречается в нескольких сплитах (max={max_splits})"
        )
    actual = set(df["split"].unique())
    expected = {"train", "val", "test"}
    if actual != expected:
        raise ValueError(f"Неожиданные метки сплитов: {sorted(actual)}")


def summarize_split(df: pd.DataFrame) -> dict:
    stats: dict = {
        "total_samples": int(len(df)),
        "total_groups": int(df["group_id"].nunique()),
        "samples_by_split": df["split"].value_counts().to_dict(),
        "groups_by_split": df.groupby("split")["group_id"].nunique().to_dict(),
        "labels_by_split": (
            df.groupby(["split", "label"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")  # ty:ignore[no-matching-overload]
        ),
        "strategies_by_split": (
            df.groupby(["split", "strategy"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")  # ty:ignore[no-matching-overload]
        ),
        "length_stats_by_split": (
            df.groupby("split")["length"]
            .describe()
            .round(3)
            .fillna(0)
            .to_dict(orient="index")  # ty:ignore[no-matching-overload]
        ),
    }
    return stats


def print_summary(df: pd.DataFrame, stats: dict) -> None:
    print("\n=== SPLIT SUMMARY ===")
    print(f"Total samples : {stats['total_samples']}")
    print(f"Total groups  : {stats['total_groups']}")

    print("\nSamples by split:")
    print(df["split"].value_counts().sort_index())

    print("\nGroups by split:")
    print(df.groupby("split")["group_id"].nunique().sort_index())

    print("\nLabels by split:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).sort_index())

    print("\nStrategies by split:")
    print(
        df.groupby(["split", "strategy"])
        .size()
        .unstack(fill_value=0)
        .fillna(0)
        .sort_index()
    )

    print("\nFailure Modes by split (0=OK, 1-4=Errors, 5=Unknown):")
    print(
        df.groupby(["split", "failure_mode_label"])
        .size()
        .unstack(fill_value=0)
        .fillna(0)
        .sort_index()
    )

    print("\nLength stats by split:")
    print(df.groupby("split")["length"].describe().round(2).sort_index())


def main() -> None:
    for path, name in [(POS_H5, "positive"), (NEG_H5, "negative")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} H5: {os.path.abspath(path)}")

    print(f"Reading positive H5 : {os.path.abspath(POS_H5)}")
    print(f"Reading negative H5 : {os.path.abspath(NEG_H5)}")

    df = build_manifest(POS_H5, NEG_H5)
    df.to_csv(OUT_MANIFEST, index=False)

    df = assign_group_splits(df, seed=RANDOM_SEED)
    validate_split(df)

    df = df.sort_values(
        ["split", "group_id", "label", "sample_key"],
        ascending=[True, True, False, True],
    )
    df.to_csv(OUT_MANIFEST_SPLIT, index=False)

    stats = summarize_split(df)
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print_summary(df, stats)

    print("\nSaved files:")
    for path in [OUT_MANIFEST, OUT_MANIFEST_SPLIT, OUT_STATS]:
        print(f"  - {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
