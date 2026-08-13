"""
analysis_relabel.py
-------------------
Перемер всех обученных моделей под исправленной меткой (analysis_label_choice.py):
scRMSD = min по 8 последовательностям, как определено в docs/07, вместо mean.

Почему нельзя просто сравнить lift «до» и «после»: база меняется. Под mean
дизайнируемых 27.6% (RFdiffusion), под min — 70.5%, а lift арифметически
ограничен 1/base_rate. Падение lift при смене метки само по себе ничего не
говорит о модели. Поэтому здесь всё меряется ОТНОСИТЕЛЬНО потолка оракула,
измеренного на той же метке.

Потолок и модель ставятся в одинаковые условия: последовательности делятся
4 + 4, оракул ранжирует по половине A, модель — своим предсказанием, и обе
проверяются по одной и той же половине B. Раньше модель проверялась по полной
метке из 8, а оракул — по половине, что завышало долю «забранного» сигнала.

Запуск:  python analysis_relabel.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from analysis_label_choice import (
    AGGREGATORS,
    MIN_SAMPLES_PER_MOTIF,
    N_SPLITS,
    TOP_FRAC,
)
from analysis_logits_ranking import EVAL_SOURCES, GENERATOR_DIRS, parse_pdb_files
from analysis_oracle_ceiling import (
    DESIGNABLE_MAX_SCRMSD,
    SATURATED_HIGH,
    SATURATED_LOW,
    load_per_sequence_rmsd,
)
from analysis_perresidue import score
from inference import build_model
from train_soft import HOLDOUT_GENERATORS

RNG_SEED = 42
OUT_JSON = "analysis_relabel.json"


def precision_at_top(rank_by: np.ndarray, design: np.ndarray) -> float:
    k = max(1, round(TOP_FRAC * len(design)))
    return float(design[np.argsort(rank_by)[:k]].mean())


def evaluate_motif(pred: np.ndarray, samples: list[np.ndarray], agg, rng) -> dict | None:
    """Ранжирование модели против метки agg на одном мотиве.

    Возвращает и абсолютные числа против полной метки (8 seq), и честное
    сравнение с оракулом на общей половине B.
    """
    full = np.array([agg(s) for s in samples])
    design_full = full < DESIGNABLE_MAX_SCRMSD
    base = design_full.mean()
    if not (SATURATED_LOW <= base <= SATURATED_HIGH):
        return None

    sp = spearmanr(pred, full)[0] if len(np.unique(full)) > 1 else np.nan
    lift_full = precision_at_top(pred, design_full) / base

    m_lifts, o_lifts = [], []
    for _ in range(N_SPLITS):
        a, b = [], []
        for s in samples:
            idx = rng.permutation(len(s))
            half = len(s) // 2
            a.append(agg(s[idx[:half]]))
            b.append(agg(s[idx[half:]]))
        a, b = np.array(a), np.array(b)
        design_b = b < DESIGNABLE_MAX_SCRMSD
        base_b = design_b.mean()
        if base_b == 0 or base_b == 1:
            continue
        m_lifts.append(precision_at_top(pred, design_b) / base_b)
        o_lifts.append(precision_at_top(a, design_b) / base_b)

    if not m_lifts:
        return None
    return {
        "base_rate": float(base),
        "spearman": float(sp),
        "lift": float(lift_full),
        "model_lift_vs_B": float(np.mean(m_lifts)),
        "oracle_lift_vs_B": float(np.mean(o_lifts)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variants = []
    soft = build_model("checkpoints/soft_model.pth", device)
    variants.append(("soft", soft, "fold"))
    if os.path.exists("checkpoints/joint_model.pth"):
        joint = build_model("checkpoints/joint_model.pth", device, per_residue=True)
        variants.append(("joint", joint, "fold"))

    results = {}
    for gen in GENERATOR_DIRS:
        records, _ = parse_pdb_files(GENERATOR_DIRS[gen])
        per_seq = load_per_sequence_rmsd(EVAL_SOURCES[gen])

        by_motif: dict[str, list[np.ndarray]] = {}
        for (motif, sample), rmsd in per_seq.items():
            by_motif.setdefault(motif, []).append((sample, rmsd))

        results[gen] = {}
        for vname, model, readout in variants:
            pred = score(model, records, device, readout)
            lookup = dict(zip(zip(pred["motif"], pred["sample"]), pred["pred_scrmsd"]))

            for agg_name, agg in AGGREGATORS.items():
                rng = np.random.default_rng(RNG_SEED)  # общий поток на вариант
                rows = []
                for motif, items in by_motif.items():
                    pairs = [(lookup.get((motif, s)), r) for s, r in items]
                    pairs = [(p, r) for p, r in pairs if p is not None]
                    if len(pairs) < MIN_SAMPLES_PER_MOTIF:
                        continue
                    p = np.array([x[0] for x in pairs])
                    samples = [x[1] for x in pairs]
                    r = evaluate_motif(p, samples, agg, rng)
                    if r is not None:
                        rows.append({"motif": motif, "n": len(pairs), **r})

                ms = pd.DataFrame(rows)
                results[gen][f"{vname}/{agg_name}"] = {
                    "n_kept_motifs": int(len(ms)),
                    "median_spearman": float(ms["spearman"].median()) if len(ms) else None,
                    "median_lift": float(ms["lift"].median()) if len(ms) else None,
                    "median_model_lift_vs_B": float(ms["model_lift_vs_B"].median()) if len(ms) else None,
                    "median_oracle_lift_vs_B": float(ms["oracle_lift_vs_B"].median()) if len(ms) else None,
                    "frac_of_ceiling": float(
                        (ms["model_lift_vs_B"] / ms["oracle_lift_vs_B"]).median()
                    ) if len(ms) else None,
                }

    hold = set(HOLDOUT_GENERATORS)
    print(f"\n{'генератор':16} {'модель/метка':14} {'мотивов':>8} {'Spearman':>9} "
          f"{'lift':>7} {'потолок':>8} {'доля потолка':>13}")
    print("-" * 84)
    for gen, r in results.items():
        tag = "  ХОЛДАУТ" if gen in hold else ""
        print(f"{gen}{tag}")
        for key, v in r.items():
            if v["median_spearman"] is None:
                continue
            print(f"{'':16} {key:14} {v['n_kept_motifs']:>8d} "
                  f"{v['median_spearman']:>9.3f} {v['median_model_lift_vs_B']:>6.2f}x "
                  f"{v['median_oracle_lift_vs_B']:>7.2f}x {v['frac_of_ceiling']:>12.0%}")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}")


if __name__ == "__main__":
    main()
