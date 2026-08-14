import json
import os

import h5py
import numpy as np
import pandas as pd

from dataset.sequence_clusters import download_clusters, entry_groups, homology_across_groups

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

POS_H5 = os.path.join(DATA_DIR, "positive_proteins.h5")
NEG_H5 = os.path.join(DATA_DIR, "negative_proteins.h5")
CLUSTERS = os.path.join(DATA_DIR, "clusters-by-entity-30.txt")

OUT_MANIFEST = os.path.join(DATA_DIR, "manifest_v1.csv")
OUT_MANIFEST_SPLIT = os.path.join(DATA_DIR, "manifest_v1_split.csv")
OUT_STATS = os.path.join(DATA_DIR, "split_stats_v1.json")

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC вычисляется как остаток: 1 - TRAIN_FRAC - VAL_FRAC

# Исключаемые из датасета стратегии (их образцы остаются в h5, но не
# попадают в манифест и сплиты).
EXCLUDE_STRATEGIES = {
    "borderline_hinge_defect",
    "borderline_local_fragment_rotation",
}


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


def _manifest_rel_path(h5_path: str) -> str:
    """Путь к h5 относительно директории манифеста — датасет переносим."""
    out_dir = os.path.dirname(os.path.abspath(OUT_MANIFEST))
    return os.path.relpath(os.path.abspath(h5_path), out_dir)


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
                    "source_h5": _manifest_rel_path(pos_h5_path),
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
                    "source_h5": _manifest_rel_path(neg_h5_path),
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
                    "failure_mode_label": int(
                        _safe_attr(grp.attrs, "failure_mode_label", 5)
                    ),
                }
            )
    return rows


def assign_homology_groups(df: pd.DataFrame, clusters_path: str) -> pd.DataFrame:
    """Переписывает group_id с цепи на группу гомологов (кластеры RCSB 30%).

    Декои наследуют группу своего натива через parent_positive_key, поэтому
    натив и все его деформации по-прежнему неразделимы; дополнительно вместе
    едут все цепи, чья последовательность попадает в тот же кластер.
    """
    entry_to_group = entry_groups(download_clusters(clusters_path))

    natives = df[df["label"] == 1]
    entry_of_key = dict(
        zip(natives["h5_group_key"], natives["pdb_id"].astype(str).str.upper())
    )
    # Запись вне файла кластеров (свежая, ещё не кластеризованная) — сама себе группа
    group_of_key = {
        key: entry_to_group.get(entry, f"entry::{entry}")
        for key, entry in entry_of_key.items()
    }

    unknown = sum(1 for e in entry_of_key.values() if e not in entry_to_group)
    df = df.copy()
    df["group_id"] = df["parent_positive_key"].map(group_of_key)
    missing = int(df["group_id"].isna().sum())
    if missing:
        raise ValueError(f"{missing} образцов без родительского натива в манифесте")

    used = {k: v for k, v in group_of_key.items()}
    risky = homology_across_groups(
        {entry_of_key[k]: v for k, v in used.items()}, clusters_path
    )
    print(
        f"Группы гомологов: {df['group_id'].nunique():,} "
        f"(цепей {len(natives):,}, записей вне кластеров {unknown:,}); "
        f"верхняя оценка остаточной гомологии между группами {risky:.1%}"
    )
    return df


def build_manifest(pos_h5_path: str, neg_h5_path: str, clusters_path: str) -> pd.DataFrame:
    rows = read_positive_manifest(pos_h5_path) + read_negative_manifest(neg_h5_path)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Manifest is empty: no valid samples found in input H5 files.")
    df = assign_homology_groups(df, clusters_path)
    if EXCLUDE_STRATEGIES:
        before = len(df)
        df = df[~df["strategy"].isin(EXCLUDE_STRATEGIES)].reset_index(drop=True)
        print(
            f"Исключены стратегии {sorted(EXCLUDE_STRATEGIES)}: "
            f"{before - len(df)} образцов"
        )
    return df


def assign_group_splits(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Раскладывает группы гомологов по сплитам, целясь в доли ОБРАЗЦОВ.

    Группы теперь сильно разного размера (от одной цепи до нескольких тысяч),
    поэтому делить их поровну по счёту нельзя: доли образцов уехали бы. Группы
    перемешиваются с фиксированным сидом, идут от больших к меньшим и каждая
    попадает в сплит, который дальше всех от своей цели, — жадная упаковка,
    детерминированная и без пересечений по построению.
    """
    sizes = df.groupby("group_id").size()
    groups = sorted(sizes.index.tolist())

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda g: -int(sizes[g]))

    targets = {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": 1.0 - TRAIN_FRAC - VAL_FRAC}
    filled = {"train": 0, "val": 0, "test": 0}
    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    total = int(sizes.sum())
    for group in groups:
        split = min(targets, key=lambda s: filled[s] / total - targets[s])
        assigned[split].append(group)
        filled[split] += int(sizes[group])

    train_groups = set(assigned["train"])
    val_groups = set(assigned["val"])
    test_groups = set(assigned["test"])

    n = len(groups)
    n_test = len(test_groups)

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
        raise ValueError(f"Тестовый сплит пустой (групп всего {n})")

    split_map = {g: "train" for g in train_groups}
    split_map.update({g: "val" for g in val_groups})
    split_map.update({g: "test" for g in test_groups})

    print(
        "Доли образцов по сплитам: "
        + ", ".join(f"{s} {filled[s] / total:.3f} (цель {targets[s]:.2f}, групп {len(assigned[s]):,})"
                    for s in ("train", "val", "test"))
    )

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

    df = build_manifest(POS_H5, NEG_H5, CLUSTERS)
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
