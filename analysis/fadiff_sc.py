"""
analysis/fadiff_sc.py
---------------------
Self-consistency метрики для выходов собственного генератора (fadiff-fork
probe): для каждого таргета 8 ESMFold-рефолдов спроектированных
последовательностей выравниваются (Kabsch) на сгенерированный скаффолд,
считается CA-RMSD — полный, по мотиву и по скаффолд-областям
(позиции мотива берутся из fixed.jsonl пробы).

Затем P(fold) наших моделей (evaluation/eval_generated.py --dirs fadiff=...)
коррелирует с полученным scRMSD.

Запуск:  python -m analysis.fadiff_sc
"""

import glob
import json
import os

import biotite.structure.io.pdb as pdb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROBE_ROOT = "/home/pc/PycharmProjects/fadiff-fork/.cache/probe_numt500"
PFOLD_CSV = "results/eval_results_fadiff_detail.csv"
OUT_CSV = "results/fadiff_self_consistency.csv"
OUT_JSON = "results/eval_results_fadiff_scrmsd.json"
FIG_DIR = "figures"


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD после оптимального наложения P на Q (Kabsch)."""
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    C = P.T @ Q
    V, _, Wt = np.linalg.svd(C)
    d = np.linalg.det(Wt.T @ V.T) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    U = Wt.T @ V.T
    R = P @ U
    return float(np.sqrt(np.mean(np.sum((R - Q) ** 2, axis=-1))))


def get_ca(path: str) -> np.ndarray:
    structure = pdb.PDBFile.read(path).get_structure(model=1)
    ca = structure[structure.atom_name == "CA"]
    order = np.argsort(ca.res_id, kind="stable")
    return ca.coord[order]


def motif_positions(target_dir: str) -> set[int]:
    fixed_path = os.path.join(target_dir, "fixed.jsonl")
    if not os.path.exists(fixed_path):
        return set()
    with open(fixed_path) as fp:
        data = json.loads(fp.readline())
    positions: set[int] = set()
    for pos_list in data.get("generated", {}).values():
        positions.update(pos_list)
    return positions


def compute_self_consistency() -> pd.DataFrame:
    rows = []
    for target in sorted(os.listdir(PROBE_ROOT)):
        tdir = os.path.join(PROBE_ROOT, target)
        gen_path = os.path.join(tdir, "generated.pdb")
        esmf_files = sorted(glob.glob(os.path.join(tdir, "esmf", "sample_*.pdb")))
        if not os.path.exists(gen_path) or not esmf_files:
            continue

        ca_gen = get_ca(gen_path)
        L = len(ca_gen)
        motif = sorted(motif_positions(tdir))
        motif_mask = np.array([r in set(motif) for r in range(L)])

        rmsd_total, rmsd_motif, rmsd_scaffold = [], [], []
        for ef in esmf_files:
            ca_ef = get_ca(ef)
            if len(ca_ef) != L:
                continue
            rmsd_total.append(kabsch_rmsd(ca_ef, ca_gen))
            if motif_mask.sum() >= 3:
                rmsd_motif.append(kabsch_rmsd(ca_ef[motif_mask], ca_gen[motif_mask]))
            if (~motif_mask).sum() >= 3:
                rmsd_scaffold.append(kabsch_rmsd(ca_ef[~motif_mask], ca_gen[~motif_mask]))

        if not rmsd_total:
            continue
        rows.append(
            {
                "target": target,
                "length": L,
                "n_motif": int(motif_mask.sum()),
                "n_esmf": len(rmsd_total),
                "sc_rmsd_total": float(np.mean(rmsd_total)),
                "sc_rmsd_motif": float(np.mean(rmsd_motif)) if rmsd_motif else float("nan"),
                "sc_rmsd_scaffold": float(np.mean(rmsd_scaffold)) if rmsd_scaffold else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main():
    sc = compute_self_consistency()
    sc.to_csv(OUT_CSV, index=False)
    print(f"Self-consistency по {len(sc)} таргетам сохранена в {OUT_CSV}")
    print(
        f"scRMSD total: mean={sc['sc_rmsd_total'].mean():.2f} Å, "
        f"median={sc['sc_rmsd_total'].median():.2f}, "
        f"доля <2Å = {(sc['sc_rmsd_total'] < 2).mean():.3f}"
    )
    print(
        f"scRMSD motif: mean={sc['sc_rmsd_motif'].mean():.2f} Å | "
        f"scaffold: mean={sc['sc_rmsd_scaffold'].mean():.2f} Å"
    )

    if not os.path.exists(PFOLD_CSV):
        print(
            f"\n{PFOLD_CSV} не найден — сначала запустите:\n"
            "  python -m evaluation.eval_generated --dirs fadiff=" + PROBE_ROOT
            + " --pattern '**/generated.pdb' -o results/eval_results_fadiff.json"
        )
        return

    detail = pd.read_csv(PFOLD_CSV)
    detail["target"] = detail["motif"]
    merged = detail.merge(
        sc.drop(columns=["length"]), on="target", how="inner"
    )
    print(f"\nСмержено: {len(merged)} строк ({merged['model'].nunique()} модели × {len(sc)} таргетов)")

    results = {"n_targets": int(len(sc)), "correlations": {}}
    print("\nКорреляции P(fold) ↔ scRMSD (n={})".format(len(sc)))
    print("| Модель | Spearman total | Spearman motif | Spearman scaffold |")
    for model, g in merged.groupby("model"):
        m = g.dropna(subset=["sc_rmsd_total"])
        sp, p = spearmanr(m["p_fold"], m["sc_rmsd_total"])
        pe, _ = pearsonr(m["p_fold"], m["sc_rmsd_total"])
        sp_m, _ = spearmanr(m["p_fold"], m["sc_rmsd_motif"])
        sp_s, _ = spearmanr(m["p_fold"], m["sc_rmsd_scaffold"])
        results["correlations"][model] = {
            "spearman_total": float(sp),
            "spearman_total_p": float(p),
            "pearson_total": float(pe),
            "spearman_motif": float(sp_m),
            "spearman_scaffold": float(sp_s),
        }
        print(f"| {model} | {sp:.3f} (p={p:.2e}) | {sp_m:.3f} | {sp_s:.3f} |")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}")

    # Графики P(fold) vs scRMSD total по моделям
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    models = ["hybrid", "gps", "mlp"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, model in zip(axes, models):
        g = merged[merged["model"] == model]
        ax.scatter(g["p_fold"], g["sc_rmsd_total"], s=22, alpha=0.75)
        sp, _ = spearmanr(g["p_fold"], g["sc_rmsd_total"])
        ax.set_title(f"{model} (Spearman = {sp:.3f})")
        ax.set_xlabel("P(fold)")
        ax.set_xlim(-0.02, 1.02)
    axes[0].set_ylabel("scRMSD total, Å")
    fig.suptitle("fadiff-fork: P(fold) vs self-consistency RMSD")
    path = os.path.join(FIG_DIR, "pfold_vs_scrmsd_fadiff.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"График сохранён: {path}")

    # Построчная таблица (таргет × модели)
    pivot = merged.pivot_table(
        index=["target", "length", "sc_rmsd_total"], columns="model", values="p_fold"
    ).reset_index()
    pivot = pivot.sort_values("sc_rmsd_total")
    print("\nПо таргетам (отсортировано по scRMSD):")
    cols = ["target", "length", "sc_rmsd_total"] + [m for m in models if m in pivot.columns]
    print(pivot[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
