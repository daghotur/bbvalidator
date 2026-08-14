"""
analysis/logits_ranking.py
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

Запуск:  python -m analysis.logits_ranking
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common.motifbench import EVAL_SOURCES, GENERATOR_DIRS, scaffold_table
from common.structures import iter_padded_batches, parse_pdb_files
from evaluation.eval_model import CHECKPOINTS as MODELS
from evaluation.eval_model import build_eval_model
from inference import _autocast_ctx

OUT_CSV = "results/analysis_logits_ranking.csv"
OUT_JSON = "results/analysis_logits_ranking.json"


@torch.no_grad()
def score_records(model, records: list[dict], device: torch.device,
                  batch_size: int = 32) -> list[dict]:
    results = [None] * len(records)
    done = 0

    for n_batch, (idxs, coords, mask) in enumerate(
        iter_padded_batches(records, batch_size, device)
    ):
        with _autocast_ctx(device):
            preds = model(coords, mask)

        fold_logit = preds["fold_logit"].float().cpu().numpy()
        rmsd_raw = preds["rmsd"].float().cpu().numpy()
        p_fold = 1.0 / (1.0 + np.exp(-fold_logit))

        for b, idx in enumerate(idxs):
            r = dict(records[idx])
            r["p_fold"] = float(p_fold[b])
            r["fold_logit"] = float(fold_logit[b])
            r["rmsd_head"] = float(rmsd_raw[b])
            r["rmsd_pred"] = float(max(0.0, np.expm1(rmsd_raw[b])))
            results[idx] = r

        done += len(idxs)
        if n_batch % 20 == 0:
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
        mb = scaffold_table(root)[["motif", "sample", "sc_rmsd"]]
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
