"""
analysis/motif_bias.py
-----------------------
Проверяет, не является ли высокая точность верхних топ-K пулового скрининга
(filter_designability.py, раздел docs/08) артефактом одного насыщенного мотива
MotifBench, а не тонкой внутримотивной дискриминацией ранжирующей модели.

Для каждого генератора и каждого из 30 мотивов MotifBench считает:
  * base_rate — доля истинно дизайнуемых (scRMSD < 2 Å) сэмплов мотива;
  * within-motif Spearman(pred_scrmsd, true_scrmsd) — есть ли ранжирующий
    сигнал внутри мотива, если различать вообще есть что;
  * lift = precision@top-10%-within-motif / base_rate — подъём над базой
    при ранжировании только внутри мотива.

Мотивы с base_rate > 0.90 или < 0.10 исключаются из headline-агрегата: при
такой базе lift статистически не определён содержательно (почти нечего или
почти всё дизайнируемо в пределах мотива).

Прогон модели детерминирован не до бита (CUDA/autocast: чек-сумма pred_scrmsd
на одних и тех же входах отличается между вызовами). Для маленьких kept-групп
(RFdiffusion-AA, EvoDiff) единичный прогон даёт заметный разброс медианного
lift (±0.15-0.2 на 5 прогонах) — поэтому pred_scrmsd усредняется по REPEATS
прогонам перед агрегацией, а не берётся из одного forward pass.

Запуск:  python -m analysis.motif_bias
"""

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common.motifbench import (
    DESIGNABLE_MAX_SCRMSD,
    EVAL_SOURCES,
    GENERATOR_DIRS,
    TRAIN_GENERATORS,
    motif_role,
    motif_split,
    scaffold_table,
)
from common.ranking import RNG_SEED, SATURATED_HIGH, SATURATED_LOW, precision_at_top
from common.scoring import score_designability
from common.structures import motifs_in, parse_pdb_files
from inference import build_model

RANKING_CKPT = "checkpoints/soft_model.pth"
TOP_DEPTHS = (10, 30, 50, 100, 200, 500)
REPEATS = 8  # усреднение pred_scrmsd против прогон-к-прогону CUDA-джиттера
BOOTSTRAP_N = 1000  # resample по мотивам — вторая (не сглаживаемая усреднением) дисперсия
OUT_JSON = "results/analysis_motif_bias.json"


def per_motif_stats(merged: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for motif, g in merged.groupby("motif"):
        n = len(g)
        base_rate = g["true_design"].mean()
        prec10 = precision_at_top(
            g["pred_scrmsd"].to_numpy(), g["true_design"].to_numpy()
        )
        sp_r, sp_p = (np.nan, np.nan)
        if n >= 5 and g["true_scrmsd"].nunique() > 1 and g["pred_scrmsd"].nunique() > 1:
            sp_r, sp_p = spearmanr(g["pred_scrmsd"], g["true_scrmsd"])
        stats.append({
            "motif": motif, "n": n, "base_rate": base_rate, "prec_top10pct": prec10,
            "spearman": sp_r, "spearman_p": sp_p,
        })
    ms = pd.DataFrame(stats)
    ms["lift"] = ms["prec_top10pct"] / ms["base_rate"].replace(0, np.nan)
    return ms.sort_values("n", ascending=False)


def depth_composition(merged: pd.DataFrame) -> list[dict]:
    out = []
    for k in TOP_DEPTHS:
        top = merged.head(k)
        vc = top["motif"].value_counts()
        out.append({
            "depth": k,
            "precision": float(top["true_design"].mean()),
            "n_motifs": int(vc.shape[0]),
            "dominant_motif": vc.index[0],
            "dominant_frac": float(vc.iloc[0] / k),
        })
    return out


def bootstrap_median_lift(lifts: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    """Resample мотивов (не структур) с возвратом — вторая дисперсия, отдельная от
    прогон-к-прогону шума pred_scrmsd: сама медиана по маленькой выборке мотивов грубая,
    даже если каждый lift в ней идеально денойзен усреднением по REPEATS."""
    lifts = lifts[~np.isnan(lifts)]
    n = len(lifts)
    if n == 0:
        return {"n": 0}
    boot_medians = np.array([
        np.median(rng.choice(lifts, size=n, replace=True)) for _ in range(n_boot)
    ])
    return {
        "n": n,
        "values_sorted": sorted(round(float(x), 3) for x in lifts),
        "point_median": float(np.median(lifts)),
        "bootstrap_median": float(boot_medians.mean()),
        "iqr_low": float(np.percentile(boot_medians, 25)),
        "iqr_high": float(np.percentile(boot_medians, 75)),
        "ci90_low": float(np.percentile(boot_medians, 5)),
        "ci90_high": float(np.percentile(boot_medians, 95)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    print(f"Ранжирующая модель: {RANKING_CKPT}")
    model = build_model(RANKING_CKPT, device)
    rng = np.random.default_rng(RNG_SEED)

    all_motifs = set()
    for g in TRAIN_GENERATORS:
        all_motifs |= motifs_in(GENERATOR_DIRS[g])
    _, val_motifs = motif_split(all_motifs)

    results = {}
    for gen, scaffold_root in GENERATOR_DIRS.items():
        records, skipped = parse_pdb_files(scaffold_root)
        gt = scaffold_table(EVAL_SOURCES[gen]).rename(columns={"motif": "gt_motif"})
        runs = [score_designability(model, records, device) for _ in range(REPEATS)]
        pred = runs[0][["sample", "motif"]].copy()
        pred["pred_scrmsd"] = np.mean([r["pred_scrmsd"].to_numpy() for r in runs], axis=0)
        merged = pred.merge(
            gt[["sample", "sc_rmsd"]], on="sample", how="inner"
        ).rename(columns={"sc_rmsd": "true_scrmsd"})
        merged["true_design"] = merged["true_scrmsd"] < DESIGNABLE_MAX_SCRMSD
        merged = merged.sort_values("pred_scrmsd").reset_index(drop=True)

        ms = per_motif_stats(merged)
        saturated = ms[(ms["base_rate"] > SATURATED_HIGH) | (ms["base_rate"] < SATURATED_LOW)]
        kept = ms[(ms["base_rate"] <= SATURATED_HIGH) & (ms["base_rate"] >= SATURATED_LOW)]

        breakdown_path = f"results/motif_breakdown_{gen.replace('/', '_')}.csv"
        ms.to_csv(breakdown_path, index=False)

        boot = bootstrap_median_lift(kept["lift"].to_numpy(), BOOTSTRAP_N, rng)

        summary = {
            "generator": gen,
            "n_samples": int(len(merged)),
            "n_motifs": int(ms.shape[0]),
            "base_rate_min": float(ms["base_rate"].min()),
            "base_rate_max": float(ms["base_rate"].max()),
            "base_rate_std": float(ms["base_rate"].std()),
            "n_saturated_excluded": int(len(saturated)),
            "saturated_motifs": saturated["motif"].tolist(),
            "n_kept": int(len(kept)),
            "median_lift_kept": float(kept["lift"].median()) if len(kept) else None,
            "mean_lift_kept": float(kept["lift"].mean()) if len(kept) else None,
            "median_spearman_kept": float(kept["spearman"].median()) if len(kept) else None,
            "n_significant_kept": int((kept["spearman_p"] < 0.05).sum()),
            "bootstrap_lift": boot,
            "depth_composition": depth_composition(merged),
            "breakdown_csv": breakdown_path,
            # in-sample против невиданного: три генератора из пяти участвовали
            # в дообучении, и их lift — не обобщение
            "median_lift_by_role": {
                role: float(sub["lift"].median())
                for role, sub in kept.assign(
                    role=[motif_role(gen, m, val_motifs) for m in kept["motif"]]
                ).groupby("role")
                if len(sub)
            } if len(kept) else {},
        }
        results[gen] = summary

        print(f"\n=== {gen} ===")
        print(f"  {summary['n_motifs']} мотивов, base_rate [{summary['base_rate_min']:.2f}, "
              f"{summary['base_rate_max']:.2f}], исключено насыщенных: {summary['n_saturated_excluded']}")
        if summary["n_kept"]:
            print(f"  median lift (n={summary['n_kept']}) = {summary['median_lift_kept']:.2f}x, "
                  f"median Spearman = {summary['median_spearman_kept']:.3f}, "
                  f"значимых p<0.05: {summary['n_significant_kept']}/{summary['n_kept']}")
            for role, lift in sorted(summary["median_lift_by_role"].items()):
                print(f"    {role:11}: median lift = {lift:.2f}x")
            print(f"  bootstrap ({BOOTSTRAP_N}x, по мотивам): median={boot['bootstrap_median']:.2f}x, "
                  f"IQR=[{boot['iqr_low']:.2f}, {boot['iqr_high']:.2f}], "
                  f"90% CI=[{boot['ci90_low']:.2f}, {boot['ci90_high']:.2f}]")
        else:
            print("  недостаточно немасыщенных мотивов для headline-агрегата")
        top100 = summary["depth_composition"][3]
        assert top100["depth"] == 100
        print(f"  топ-100: precision={top100['precision']:.3f}, "
              f"доминант={top100['dominant_motif']} ({top100['dominant_frac']:.0%})")
        print(f"  breakdown: {breakdown_path}")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}")


if __name__ == "__main__":
    main()
