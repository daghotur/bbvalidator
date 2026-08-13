"""
build_lddt_labels.py
--------------------
Извлекает по-остаточную метку дизайнуемости из уже посчитанных рефолдов
MotifBench: вместо одного скаляра scRMSD на структуру — вектор длины L.

Метка = CA-lDDT сгенерированного остова против ESMFold-рефолда его же
дизайненной последовательности, усреднённый по 8 последовательностям.
Смысл: «насколько локальное окружение остатка i воспроизводится, когда
последовательность реально сворачивают». 1.0 — воспроизводится точно,
0.0 — не воспроизводится.

Почему lDDT, а не отклонение после Kabsch: lDDT не требует суперпозиции,
поэтому один развалившийся сегмент не разворачивает систему координат и не
размазывает ошибку по всей цепи (проверено: ошибка локализована, 27%
остатков дают 50% суммарного отклонения).

Плотность супервизии: ~15k структур × ~L остатков ≈ 1.5M меток вместо 15k.

Выход: lddt_labels/<generator>.npz — per-sample массивы lddt (L,), а также
std по 8 рефолдам (мера согласия оракула на каждый остаток).

Запуск:  python build_lddt_labels.py
"""

import glob
import os

import numpy as np

from analysis_logits_ranking import EVAL_SOURCES
from inference import parse_pdb_to_backbone

OUT_DIR = "lddt_labels"
INCLUSION_RADIUS = 15.0
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
MIN_SEQ_SEP = 1
MIN_REFOLDS = 4


def ca_coords(path: str) -> np.ndarray:
    return parse_pdb_to_backbone(path)[:, 1, :].astype(np.float64)


def per_residue_lddt(ref: np.ndarray, mod: np.ndarray) -> np.ndarray:
    """Стандартный CA-lDDT: доля сохранённых межостаточных расстояний вокруг
    каждого остатка, усреднённая по четырём допускам."""
    d_ref = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
    d_mod = np.linalg.norm(mod[:, None, :] - mod[None, :, :], axis=-1)
    idx = np.arange(len(ref))
    sep = np.abs(idx[:, None] - idx[None, :])
    mask = (d_ref < INCLUSION_RADIUS) & (sep >= MIN_SEQ_SEP)
    np.fill_diagonal(mask, False)

    diff = np.abs(d_ref - d_mod)
    preserved = np.zeros_like(d_ref)
    for t in THRESHOLDS:
        preserved += (diff < t).astype(np.float64)
    preserved /= len(THRESHOLDS)

    num = (preserved * mask).sum(axis=1)
    den = mask.sum(axis=1)
    return num / np.maximum(den, 1)


def process_generator(generator: str, root: str) -> dict:
    samples = sorted(glob.glob(os.path.join(root, "**", "self_consistency"), recursive=True))
    out = {}
    stats = {"ok": 0, "no_refolds": 0, "length_mismatch": 0, "parse_error": 0}

    for sc_dir in samples:
        if "__MACOSX" in sc_dir:
            continue
        sample_dir = os.path.dirname(sc_dir)
        name = os.path.basename(sample_dir)
        motif = os.path.basename(os.path.dirname(sample_dir))
        gen_pdb = os.path.join(sample_dir, f"{name}.pdb")
        refolds = sorted(glob.glob(os.path.join(sc_dir, "esmf", "sample_*.pdb")))

        if not os.path.exists(gen_pdb) or len(refolds) < MIN_REFOLDS:
            stats["no_refolds"] += 1
            continue
        try:
            ref = ca_coords(gen_pdb)
        except Exception:
            stats["parse_error"] += 1
            continue

        profiles = []
        for r in refolds:
            try:
                mod = ca_coords(r)
            except Exception:
                continue
            if len(mod) != len(ref):
                continue
            profiles.append(per_residue_lddt(ref, mod))

        if len(profiles) < MIN_REFOLDS:
            stats["length_mismatch"] += 1
            continue

        arr = np.array(profiles)
        out[f"{motif}/{name}"] = np.stack([arr.mean(0), arr.std(0)]).astype(np.float32)
        stats["ok"] += 1

    return out, stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for generator, root in EVAL_SOURCES.items():
        out, stats = process_generator(generator, root)
        path = os.path.join(OUT_DIR, f"{generator.replace('/', '_')}.npz")
        np.savez_compressed(path, **out)
        n_res = sum(v.shape[1] for v in out.values())
        print(f"{generator:16} структур {stats['ok']:5d} | остатков (меток) {n_res:8d} | "
              f"пропущено: нет рефолдов {stats['no_refolds']}, "
              f"длина не совпала {stats['length_mismatch']}, ошибка парсинга {stats['parse_error']}")
        print(f"                 → {path}")


if __name__ == "__main__":
    main()
