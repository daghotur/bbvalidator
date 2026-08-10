"""
analysis_scrmsd.py
------------------
Корреляция P(foldable) наших моделей с MotifBench self-consistency метриками
(scRMSD / scTM) по сэмплам внешних генераторов.

MotifBench для каждого скаффолда дизайнит 8 последовательностей (ProteinMPNN),
рефолдит их ESMFold и считает RMSD/TM-score рефолда против скаффолда.
scRMSD/scTM скаффолда = среднее по 8 последовательностям.

Запуск:  python analysis_scrmsd.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

EVAL_SOURCES = {
    "RFdiffusion": "data/ood/eval_rfdiffusion",
    "RFdiffusion-AA": "data/ood/eval_rfdiffusion_aa",
    "ODesign-Rigid": "data/ood/eval_odesign_rigid",
    "GPDL": "data/ood/eval_gpdl",
    "EvoDiff": "data/ood/eval_evodiff",
}
DETAIL_CSV = "eval_results_generated_detail.csv"
OUT_JSON = "eval_results_scrmsd.json"
FIG_DIR = "figures"


def parse_motifbench_eval(root: str) -> pd.DataFrame:
    """Собирает per-scaffold scRMSD/scTM из всех esm_eval_results.csv."""
    rows = []
    for csv_path in glob.glob(os.path.join(root, "**", "esm_eval_results.csv"), recursive=True):
        if "__MACOSX" in csv_path:
            continue
        parts = csv_path.split(os.sep)
        # .../<motif>/<sample>/self_consistency/esm_eval_results.csv
        sample = parts[-3]
        motif = parts[-4]
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "rmsd" not in df.columns or "tm_score" not in df.columns:
            continue
        rmsd = pd.to_numeric(df["rmsd"], errors="coerce").dropna()
        tm = pd.to_numeric(df["tm_score"], errors="coerce").dropna()
        if len(rmsd) == 0:
            continue
        rows.append(
            {
                "motif": motif,
                "sample": sample,
                "sc_rmsd": float(rmsd.mean()),
                "sc_tm": float(tm.mean()),
                "n_seqs": int(len(rmsd)),
            }
        )
    return pd.DataFrame(rows)


def correlations(merged: pd.DataFrame) -> dict:
    out = {}
    for model, g in merged.groupby("model"):
        m = g.dropna(subset=["sc_rmsd", "p_fold"])
        sp_r, sp_p = spearmanr(m["p_fold"], m["sc_rmsd"])
        pe_r, pe_p = pearsonr(m["p_fold"], m["sc_rmsd"])
        sp_tm, _ = spearmanr(m["p_fold"], m["sc_tm"])
        out[model] = {
            "n": int(len(m)),
            "spearman_pfold_vs_scRMSD": float(sp_r),
            "spearman_p": float(sp_p),
            "pearson_pfold_vs_scRMSD": float(pe_r),
            "pearson_p": float(pe_p),
            "spearman_pfold_vs_scTM": float(sp_tm),
        }
    return out


def binned_means(merged: pd.DataFrame, n_bins: int = 5) -> dict:
    out = {}
    for model, g in merged.groupby("model"):
        m = g.dropna(subset=["sc_rmsd", "p_fold"])
        m = m.assign(bin=pd.qcut(m["p_fold"], n_bins, duplicates="drop"))
        agg = m.groupby("bin", observed=True).agg(
            n=("sc_rmsd", "size"),
            sc_rmsd_mean=("sc_rmsd", "mean"),
            sc_tm_mean=("sc_tm", "mean"),
        )
        out[model] = [
            {
                "p_fold_range": f"{idx.left:.3f}-{idx.right:.3f}",
                "n": int(r["n"]),
                "sc_rmsd_mean": float(r["sc_rmsd_mean"]),
                "sc_tm_mean": float(r["sc_tm_mean"]),
            }
            for idx, r in agg.iterrows()
        ]
    return out


def make_plots(merged_by_group: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    models = ["hybrid", "gps", "mlp"]

    for group, merged in merged_by_group.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, model in zip(axes, models):
            g = merged[merged["model"] == model]
            ax.scatter(g["p_fold"], g["sc_rmsd"], s=6, alpha=0.35)
            sp, _ = spearmanr(g["p_fold"], g["sc_rmsd"])
            ax.set_title(f"{model} (Spearman = {sp:.3f})")
            ax.set_xlabel("P(fold)")
            ax.set_xlim(-0.02, 1.02)
        axes[0].set_ylabel("scRMSD, Å")
        fig.suptitle(f"{group}: P(fold) vs scRMSD (MotifBench self-consistency)")
        path = os.path.join(FIG_DIR, f"pfold_vs_scrmsd_{group.lower().replace('-', '')}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"График сохранён: {path}")


def main():
    detail = pd.read_csv(DETAIL_CSV)
    results = {}
    merged_by_group = {}

    for group, root in EVAL_SOURCES.items():
        mb = parse_motifbench_eval(root)
        print(f"{group}: {len(mb)} скаффолдов с self-consistency метриками "
              f"(scRMSD mean={mb['sc_rmsd'].mean():.2f} Å, median={mb['sc_rmsd'].median():.2f})")

        det = detail[detail["group"] == group].copy()
        det["sample"] = det["name"].str.replace(".pdb", "", regex=False)
        merged = det.merge(mb, on=["motif", "sample"], how="inner")
        print(f"  смержено с нашими оценками: {len(merged)} строк "
              f"({merged['model'].nunique()} модели × {len(merged)//3} структур)")

        results[group] = {
            "motifbench_stats": {
                "n_scaffolds": int(len(mb)),
                "sc_rmsd_mean": float(mb["sc_rmsd"].mean()),
                "sc_rmsd_median": float(mb["sc_rmsd"].median()),
                "sc_tm_mean": float(mb["sc_tm"].mean()),
                "frac_scRMSD_below_2A": float((mb["sc_rmsd"] < 2.0).mean()),
            },
            "correlations": correlations(merged),
            "p_fold_bins_vs_scRMSD": binned_means(merged),
        }
        merged_by_group[group] = merged

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {os.path.abspath(OUT_JSON)}")

    make_plots(merged_by_group)

    # Печать сводки
    for group, res in results.items():
        print(f"\n===== {group} =====")
        print("Корреляции P(fold):")
        print("| Модель | n | Spearman vs scRMSD | Pearson vs scRMSD | Spearman vs scTM |")
        for model, c in res["correlations"].items():
            print(
                f"| {model} | {c['n']} | {c['spearman_pfold_vs_scRMSD']:.3f} "
                f"(p={c['spearman_p']:.1e}) | {c['pearson_pfold_vs_scRMSD']:.3f} "
                f"| {c['spearman_pfold_vs_scTM']:.3f} |"
            )
        print("\nP(fold)-бины → средний scRMSD:")
        for model, rows in res["p_fold_bins_vs_scRMSD"].items():
            print(f"  {model}: " + " | ".join(
                f"{r['p_fold_range']}: {r['sc_rmsd_mean']:.2f}Å (n={r['n']})" for r in rows
            ))


if __name__ == "__main__":
    main()
