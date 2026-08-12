"""
analysis_logits_ranking.py
--------------------------
Нулевая диагностика ранжирующего сигнала на дальнем OOD: проверяет, не является
ли ослабление корреляций P(fold) ↔ scRMSD артефактом насыщения сигмоиды.

Для 18k структур MotifBench сравниваются Spearman(scRMSD) трёх выходов моделей:
  (a) P(fold) = sigmoid(fold_logit) — как в eval_results_generated_detail.csv;
  (b) сырой fold_logit до сигмоиды;
  (c) вспомогательная rmsd-голова (непрерывный выход, пространство log1p).

Если логит или rmsd-голова ранжируют заметно лучше P(fold), часть «потери
ранжирования» — артефакт динамического диапазона выхода, а не утрата
информации представлением.

Запуск:  python analysis_logits_ranking.py
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from analysis_scrmsd import parse_motifbench_eval
from eval_model import build_eval_model

GENERATOR_DIRS = {
    "RFdiffusion": "data/ood/rfdiffusion",
    "RFdiffusion-AA": "data/ood/rfdiffusion_aa",
    "ODesign-Rigid": "data/ood/odesign_rigid",
    "GPDL": "data/ood/gpdl",
    "EvoDiff": "data/ood/evodiff",
}
EVAL_SOURCES = {
    "RFdiffusion": "data/ood/eval_rfdiffusion",
    "RFdiffusion-AA": "data/ood/eval_rfdiffusion_aa",
    "ODesign-Rigid": "data/ood/eval_odesign_rigid",
    "GPDL": "data/ood/eval_gpdl",
    "EvoDiff": "data/ood/eval_evodiff",
}
MODELS = {
    "hybrid": "checkpoints/best_model.pth",
    "mlp": "checkpoints/baseline_mlp_best.pth",
    "gps": "checkpoints/baseline_gps_best.pth",
}
OUT_CSV = "analysis_logits_ranking.csv"
OUT_JSON = "analysis_logits_ranking.json"


def parse_pdb_files(root_dir: str) -> list[dict]:
    pdb_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.pdb"), recursive=True))
    pdb_files = [
        f
        for f in pdb_files
        if "__MACOSX" not in f and not os.path.basename(f).startswith("._")
    ]

    from inference import parse_pdb_to_backbone

    records = []
    skipped = 0
    for path in pdb_files:
        try:
            coords = parse_pdb_to_backbone(path).astype(np.float32)
        except Exception:
            skipped += 1
            continue
        parts = path.split(os.sep)
        records.append({"sample": parts[-1], "motif": parts[-2], "coords": coords})
    return records, skipped


@torch.no_grad()
def score_records(model, records: list[dict], device: torch.device,
                  batch_size: int = 32) -> list[dict]:
    from inference import _autocast_ctx, center_coords

    order = sorted(range(len(records)), key=lambda i: len(records[i]["coords"]))
    results = [None] * len(records)
    done = 0

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
            preds = model(coords_t, mask_t)

        fold_logit = preds["fold_logit"].float().cpu().numpy()
        rmsd_raw = preds["rmsd"].float().cpu().numpy()
        p_fold = 1.0 / (1.0 + np.exp(-fold_logit))

        for b, idx in enumerate(batch_idx):
            r = dict(records[idx])
            r["p_fold"] = float(p_fold[b])
            r["fold_logit"] = float(fold_logit[b])
            r["rmsd_head"] = float(rmsd_raw[b])
            r["rmsd_pred"] = float(max(0.0, np.expm1(rmsd_raw[b])))
            results[idx] = r

        done += B
        if (start // batch_size) % 20 == 0:
            print(f"    скоринг: {done}/{len(records)}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Устройство: {device}")

    # Парсинг структур один раз
    records_by_group, n_skipped = {}, 0
    for group, root in GENERATOR_DIRS.items():
        records, skipped = parse_pdb_files(root)
        records_by_group[group] = records
        n_skipped += skipped
        print(f"{group}: {len(records)} структур" + (f" (пропущено {skipped})" if skipped else ""))

    # Ground truth scRMSD по генераторам
    scrmsd_by_group = {}
    for group, root in EVAL_SOURCES.items():
        mb = parse_motifbench_eval(root)[["motif", "sample", "sc_rmsd"]]
        scrmsd_by_group[group] = mb
        print(f"{group}: {len(mb)} скаффолдов со scRMSD")

    summary = {}
    all_rows = []
    for model_name, ckpt in MODELS.items():
        print(f"\n=== {model_name} ({ckpt}) ===")
        model, arch = build_eval_model(ckpt, device, pca_path="dataset/pca_components.pth")
        model.eval()

        for group, records in records_by_group.items():
            scored = score_records(model, records, device, args.batch_size)
            df = pd.DataFrame(
                [
                    {k: v for k, v in r.items() if k != "coords"} | {"group": group}
                    for r in scored
                ]
            )
            df["sample"] = df["sample"].str.replace(".pdb", "", regex=False)
            df = df.merge(scrmsd_by_group[group], on=["motif", "sample"], how="inner")
            df["model"] = model_name
            all_rows.append(df)

            m = df.dropna(subset=["sc_rmsd"])
            sp_pfold, _ = spearmanr(m["p_fold"], m["sc_rmsd"])
            sp_logit, _ = spearmanr(m["fold_logit"], m["sc_rmsd"])
            sp_rmsd, _ = spearmanr(m["rmsd_head"], m["sc_rmsd"])
            summary.setdefault(group, {})[model_name] = {
                "n": int(len(m)),
                "spearman_pfold": float(sp_pfold),
                "spearman_fold_logit": float(sp_logit),
                "spearman_rmsd_head": float(sp_rmsd),
                "frac_pfold_gt_0.99": float((m["p_fold"] > 0.99).mean()),
                "fold_logit_std": float(m["fold_logit"].std()),
                "fold_logit_q": [float(q) for q in np.quantile(m["fold_logit"], [0.05, 0.5, 0.95])],
            }
            print(f"  {group}: n={len(m)} | Spearman pfold={sp_pfold:+.3f} "
                  f"logit={sp_logit:+.3f} rmsd_head={sp_rmsd:+.3f} | "
                  f"P>0.99: {(m['p_fold'] > 0.99).mean():.1%}, logit std={m['fold_logit'].std():.2f}")

    merged = pd.concat(all_rows, ignore_index=True)
    merged.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    # Итоговая таблица
    print("\n===== Spearman ↔ scRMSD по выходам моделей =====")
    print("| Генератор | Модель | pfold | fold_logit | rmsd-голова |")
    for group, models in summary.items():
        for model_name, s in models.items():
            print(f"| {group} | {model_name} | {s['spearman_pfold']:+.3f} "
                  f"| {s['spearman_fold_logit']:+.3f} | {s['spearman_rmsd_head']:+.3f} |")

    print(f"\nСохранено: {os.path.abspath(OUT_CSV)}, {os.path.abspath(OUT_JSON)}")
    if n_skipped:
        print(f"Пропущено при парсинге: {n_skipped}")


if __name__ == "__main__":
    main()
