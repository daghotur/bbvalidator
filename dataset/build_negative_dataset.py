import os
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

INPUT_H5 = "positive_proteins.h5"
OUTPUT_H5 = "negative_proteins.h5"
RANDOM_SEED = 42
N_DECOYS_PER_POSITIVE = 2
MIN_SPLIT_GAP = 15

# Маппинг стратегий в классы для Aux Head
FAILURE_MODE_MAP = {
    "positive_real": 0,
    "easy_global_noise": 1,
    "easy_chain_break": 1,
    "hard_core_unpacked": 2,
    "hard_false_compact": 2,
    "hard_near_native": 3,
    "borderline_hinge_defect": 4,
    "borderline_local_fragment_rotation": 4,
    "unknown_negative": 5,
}

# steric_target проставляет dataset/compute_targets.py
# отдельным проходом после генерации негативов.

# ── helpers ──────────────────────────────────────────────────────────────────


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    # Берем только C-alpha (индекс 1 во второй оси)
    ca1 = coords1[:, 1, :]
    ca2 = coords2[:, 1, :]

    # Центрируем
    ca1_c = ca1 - ca1.mean(axis=0)
    ca2_c = ca2 - ca2.mean(axis=0)

    # Ковариационная матрица
    H = ca1_c.T @ ca2_c
    U, S, Vt = np.linalg.svd(H)

    # Защита от отражений (хиральности)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    W = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    # Оптимальная матрица вращения
    R = Vt.T @ W @ U.T

    # Выравниваем вторую структуру и считаем RMSD
    ca2_aligned = ca2_c @ R.T
    diff = ca1_c - ca2_aligned
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=-1)))
    return float(rmsd)


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _rotation_matrix(
    rng: np.random.Generator,
    axis: np.ndarray | None = None,
    angle_deg: float | None = None,
) -> np.ndarray:
    if axis is None:
        axis = _random_unit_vector(rng)
    else:
        axis = np.asarray(axis, dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    if angle_deg is None:
        angle_deg = float(rng.uniform(8.0, 25.0))
    return (
        Rotation.from_rotvec(np.radians(angle_deg) * axis)
        .as_matrix()
        .astype(np.float32)
    )


def _split_into_blocks(
    coords: np.ndarray,
    n_blocks: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n = len(coords)
    if n_blocks < 2 or n < (n_blocks * MIN_SPLIT_GAP):
        return [coords]

    cut_pool = np.arange(MIN_SPLIT_GAP, n - MIN_SPLIT_GAP)
    if len(cut_pool) < n_blocks - 1:
        return [coords]

    for _ in range(50):
        cuts = np.sort(rng.choice(cut_pool, size=n_blocks - 1, replace=False))
        boundaries = np.concatenate([[0], cuts, [n]])
        lengths = np.diff(boundaries)
        if np.all(lengths >= MIN_SPLIT_GAP):
            return [
                coords[boundaries[i] : boundaries[i + 1]]
                for i in range(len(boundaries) - 1)
            ]

    return [coords]


# ── easy negatives (20 %) ─────────────────────────────────────────────────────


def make_easy_negative(
    coords: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    c = coords.copy().astype(np.float32)
    # 50 % — global Gaussian noise; 50 % — chain break via translation
    if rng.random() > 0.5:
        c += rng.normal(0.0, 1.5, size=c.shape).astype(np.float32)
        return c, "easy_global_noise"

    cut_idx = int(rng.integers(10, len(c) - 10))
    shift = rng.uniform(12.0, 18.0, size=3).astype(np.float32)
    c[cut_idx:] += shift
    return c, "easy_chain_break"


# ── hard negatives (50 %) ─────────────────────────────────────────────────────


def make_hard_unpacking(
    coords: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    c = coords.copy().astype(np.float32)
    centroid = c.mean(axis=(0, 1), keepdims=True)
    expansion = float(rng.uniform(0.10, 0.25))
    c += (c - centroid) * expansion
    return c, "hard_core_unpacked"


def make_hard_compact_wrong_topology(
    coords: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    n = len(coords)
    if n < 60:
        return make_hard_unpacking(coords, rng)

    n_blocks = int(rng.integers(2, 5))
    blocks = _split_into_blocks(coords, n_blocks, rng)
    if len(blocks) == 1:
        return make_hard_unpacking(coords, rng)

    rng.shuffle(blocks)
    packed_blocks = []
    placement_radius = float(rng.uniform(4.5, 8.5))

    for block in blocks:
        b = block.copy().astype(np.float32)
        b -= b.mean(axis=(0, 1), keepdims=True)

        rot = _rotation_matrix(rng, angle_deg=float(rng.uniform(20.0, 120.0)))
        b = b.reshape(-1, 3) @ rot.T
        b = b.reshape(block.shape)

        direction = _random_unit_vector(rng)
        distance = float(rng.uniform(0.4, 1.0) * placement_radius)
        b += (direction * distance).reshape(1, 1, 3)
        packed_blocks.append(b)

    return (
        np.concatenate(packed_blocks, axis=0).astype(np.float32),
        "hard_false_compact",
    )


def make_hard_near_native(
    coords: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    c = coords.copy().astype(np.float32)
    centroid = c.mean(axis=(0, 1), keepdims=True)
    c -= centroid

    pre_rot = _rotation_matrix(rng, angle_deg=float(rng.uniform(0.0, 180.0)))
    post_rot = _rotation_matrix(rng, angle_deg=float(rng.uniform(0.0, 180.0)))

    flat = c.reshape(-1, 3) @ pre_rot.T
    scales = np.array(
        [rng.uniform(1.02, 1.05), rng.uniform(0.95, 0.98), rng.uniform(0.98, 1.02)],
        dtype=np.float32,
    )
    flat = flat * scales.reshape(1, 3) @ post_rot.T

    c = flat.reshape(coords.shape) + centroid
    return c.astype(np.float32), "hard_near_native"


# ── borderline negatives (30 %) ───────────────────────────────────────────────


def make_borderline_hinge(
    coords: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    c = coords.copy().astype(np.float32)
    pivot_idx = int(rng.integers(10, len(c) - 10))
    rot = _rotation_matrix(rng, angle_deg=float(rng.uniform(8.0, 22.0)))
    pivot_point = c[pivot_idx, 1, :].copy()

    tail = (c[pivot_idx:] - pivot_point).reshape(-1, 3) @ rot.T
    c[pivot_idx:] = tail.reshape(c[pivot_idx:].shape) + pivot_point
    return c.astype(np.float32), "borderline_hinge_defect"


def make_borderline_local_fragment_rotation(
    coords: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    c = coords.copy().astype(np.float32)
    n = len(c)
    window = int(rng.integers(3, 9))
    if n <= window + 2:
        return make_borderline_hinge(coords, rng)

    start = int(rng.integers(1, n - window - 1))
    end = start + window

    axis = c[end - 1, 1, :] - c[start, 1, :]
    axis_norm = np.linalg.norm(axis)
    axis = _random_unit_vector(rng) if axis_norm < 1e-6 else axis / axis_norm

    pivot = c[start:end, 1, :].mean(axis=0)
    rot = _rotation_matrix(rng, axis=axis, angle_deg=float(rng.uniform(6.0, 18.0)))

    frag = (c[start:end] - pivot).reshape(-1, 3) @ rot.T
    c[start:end] = frag.reshape(c[start:end].shape) + pivot
    return c.astype(np.float32), "borderline_local_fragment_rotation"


# ── strategy dispatcher ───────────────────────────────────────────────────────


def apply_decoy_strategy(
    coords: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    p = rng.random()

    if p < 0.20:
        return make_easy_negative(coords, rng)

    if p < 0.70:
        sub_p = rng.random()
        if sub_p < 1 / 3:
            return make_hard_unpacking(coords, rng)
        if sub_p < 2 / 3:
            return make_hard_compact_wrong_topology(coords, rng)
        return make_hard_near_native(coords, rng)

    if rng.random() > 0.5:
        return make_borderline_hinge(coords, rng)
    return make_borderline_local_fragment_rotation(coords, rng)


# ── main ──────────────────────────────────────────────────────────────────────


def build_decoys(
    input_h5: str = INPUT_H5,
    output_h5: str = OUTPUT_H5,
    n_decoys: int = N_DECOYS_PER_POSITIVE,
    seed: int = RANDOM_SEED,
) -> None:
    print(f"Генерация декоев из {input_h5} (seed={seed})...")
    print(f"Рабочая директория: {os.getcwd()}")

    if not os.path.exists(input_h5):
        raise FileNotFoundError(f"Не найден входной файл: {os.path.abspath(input_h5)}")

    rng = np.random.default_rng(seed)

    with h5py.File(input_h5, "r") as h5_in, h5py.File(output_h5, "w") as h5_out:
        keys = list(h5_in.keys())

        h5_out.attrs["random_seed"] = seed
        h5_out.attrs["source_file"] = os.path.abspath(input_h5)
        h5_out.attrs["decoys_per_positive"] = n_decoys

        counts: dict[str, int] = {}
        total = 0

        for key in tqdm(keys):
            if "coords" not in h5_in[key]:
                continue

            original_coords = h5_in[key]["coords"][:].astype(np.float32)

            for i in range(n_decoys):
                decoy_coords, strategy_name = apply_decoy_strategy(original_coords, rng)

                rmsd_val = calculate_rmsd(original_coords, decoy_coords)

                failure_label = FAILURE_MODE_MAP.get(strategy_name, 5)

                grp = h5_out.create_group(f"decoy_{i}_{key}")
                grp.create_dataset("coords", data=decoy_coords, compression="gzip")
                grp.create_dataset("label", data=np.array([0.0], dtype=np.float32))

                grp.attrs["length"] = len(decoy_coords)
                grp.attrs["strategy"] = strategy_name
                grp.attrs["source_positive_key"] = key
                grp.attrs["decoy_index"] = i
                grp.attrs["random_seed"] = seed

                # Новые Multi-Task таргеты
                grp.attrs["rmsd_target"] = rmsd_val
                grp.attrs["steric_target"] = float("nan")
                grp.attrs["failure_mode_label"] = failure_label

                counts[strategy_name] = counts.get(strategy_name, 0) + 1
                total += 1

    print("\nСтатистика сгенерированных негативов:")
    for strat, count in sorted(counts.items()):
        print(f"  - {strat}: {count} ({count / total * 100:.1f}%)")
    print(f"Всего негативов: {total}")
    print(f"Файл сохранён в: {os.path.abspath(output_h5)}")


if __name__ == "__main__":
    build_decoys()
