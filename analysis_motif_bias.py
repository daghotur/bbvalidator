"""
analysis_motif_bias.py
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

Запуск:  python analysis_motif_bias.py
"""

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from analysis_logits_ranking import EVAL_SOURCES, GENERATOR_DIRS, parse_pdb_files
from analysis_scrmsd import parse_motifbench_eval
from inference import _autocast_ctx, build_model, center_coords

RANKING_CKPT = "checkpoints/soft_model.pth"
SATURATED_HIGH = 0.90
SATURATED_LOW = 0.10
TOP_DEPTHS = (10, 30, 50, 100, 200, 500)
REPEATS = 8  # усреднение pred_scrmsd против прогон-к-прогону CUDA-джиттера
BOOTSTRAP_N = 1000  # resample по мотивам — вторая (не сглаживаемая усреднением) дисперсия
RNG_SEED = 42
OUT_JSON = "analysis_motif_bias.json"


@torch.no_grad()
def score_pred_scrmsd(model, records: list[dict], device: torch.device, batch_size: int = 32) -> pd.DataFrame:
    """Ранжирующая модель (pure soft): pred_scrmsd = expm1(fold_logit), как в filter_designability.py."""
    order = sorted(range(len(records)), key=lambda i: len(records[i]["coords"]))
    rows = [None] * len(records)
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        Lmax = max(len(records[i]["coords"]) for i in batch_idx)
        B = len(batch_idx)
        coords = np.zeros((B, Lmax, 3, 3), dtype=np.float32)
        mask = np.zeros((B, Lmax), dtype=np.bool_)
        for b, idx in enumerate(batch_idx):
            L = len(records[idx]["coords"])
            coords[b, :L] = center_coords(records[idx]["coords"])
            mask[b, :L] = True
        coords_t = torch.from_numpy(coords).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        with _autocast_ctx(device):
            fold_logit = model(coords_t, mask_t)["fold_logit"].float()
        pred_scrmsd = torch.clamp(torch.expm1(fold_logit), min=0.0).cpu().numpy()
        for b, idx in enumerate(batch_idx):
            rows[idx] = {
                "sample": records[idx]["sample"].replace(".pdb", ""),
                "motif": records[idx]["motif"],
                "pred_scrmsd": float(pred_scrmsd[b]),
            }
    return pd.DataFrame(rows)


def per_motif_stats(merged: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for motif, g in merged.groupby("motif"):
        n = len(g)
        base_rate = g["true_design"].mean()
        gs = g.sort_values("pred_scrmsd")
        k10 = max(1, round(0.1 * n))
        prec10 = gs.head(k10)["true_design"].mean()
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

    results = {}
    for gen, scaffold_root in GENERATOR_DIRS.items():
        records, skipped = parse_pdb_files(scaffold_root)
        gt = parse_motifbench_eval(EVAL_SOURCES[gen]).rename(columns={"motif": "gt_motif"})
        runs = [score_pred_scrmsd(model, records, device) for _ in range(REPEATS)]
        pred = runs[0][["sample", "motif"]].copy()
        pred["pred_scrmsd"] = np.mean([r["pred_scrmsd"].to_numpy() for r in runs], axis=0)
        merged = pred.merge(
            gt[["sample", "sc_rmsd"]], on="sample", how="inner"
        ).rename(columns={"sc_rmsd": "true_scrmsd"})
        merged["true_design"] = merged["true_scrmsd"] < 2.0
        merged = merged.sort_values("pred_scrmsd").reset_index(drop=True)

        ms = per_motif_stats(merged)
        saturated = ms[(ms["base_rate"] > SATURATED_HIGH) | (ms["base_rate"] < SATURATED_LOW)]
        kept = ms[(ms["base_rate"] <= SATURATED_HIGH) & (ms["base_rate"] >= SATURATED_LOW)]

        breakdown_path = f"motif_breakdown_{gen.replace('/', '_')}.csv"
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
        }
        results[gen] = summary

        print(f"\n=== {gen} ===")
        print(f"  {summary['n_motifs']} мотивов, base_rate [{summary['base_rate_min']:.2f}, "
              f"{summary['base_rate_max']:.2f}], исключено насыщенных: {summary['n_saturated_excluded']}")
        if summary["n_kept"]:
            print(f"  median lift (n={summary['n_kept']}) = {summary['median_lift_kept']:.2f}x, "
                  f"median Spearman = {summary['median_spearman_kept']:.3f}, "
                  f"значимых p<0.05: {summary['n_significant_kept']}/{summary['n_kept']}")
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
