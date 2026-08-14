"""
analysis/perresidue.py
----------------------
Сравнивает три режима супервизии по одной метрике — внутримотивному
ранжированию (та же, что в docs/08):

  soft   — один скаляр log1p(scRMSD) на структуру (текущий продакшн-ранжировщик);
  perres — только по-остаточный lDDT (~140 меток на структуру вместо одной);
  joint  — оба лосса вместе; у него два возможных выхода для ранжирования:
             joint/fold — голова fold_logit (прямая супервизия скаляром),
             joint/lddt — агрегат mean sigmoid(lddt_logit).

Все три стартуют из одного чекпоинта (best_model.pth) и видят одни и те же
структуры, поэтому разница объясняется разрешением метки, а не данными.
ODesign-Rigid и GPDL — холдаут, не участвовали в обучении ни одной модели.

Опорная точка — потолок оракула (analysis/oracle_ceiling.py): сколько
ранжирующего сигнала вообще есть в метке.

Запуск:  python -m analysis.perresidue
"""

import json
import os

import pandas as pd
import torch

from analysis.motif_bias import per_motif_stats
from common.motifbench import EVAL_SOURCES, GENERATOR_DIRS, SCRMSD_AGG, scaffold_table
from common.ranking import SATURATED_HIGH, SATURATED_LOW
from common.scoring import score_designability
from common.structures import parse_pdb_files
from inference import build_model
from common.motifbench import HOLDOUT_GENERATORS

OUT_JSON = "results/analysis_perresidue.json"


def summarise(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    merged = (
        pred.merge(gt[["sample", "sc_rmsd"]], on="sample", how="inner")
        .rename(columns={"sc_rmsd": "true_scrmsd"})
    )
    merged["true_design"] = merged["true_scrmsd"] < 2.0
    ms = per_motif_stats(merged)
    kept = ms[(ms["base_rate"] <= SATURATED_HIGH) & (ms["base_rate"] >= SATURATED_LOW)]
    return {
        "n": int(len(merged)),
        "n_kept_motifs": int(len(kept)),
        "median_spearman": float(kept["spearman"].median()) if len(kept) else None,
        "median_lift": float(kept["lift"].median()) if len(kept) else None,
        "n_significant": int((kept["spearman_p"] < 0.05).sum()),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Потолок берётся из relabel: там модель и оракул меряются ОДНОЙ меткой на
    # общей половине B. Потолок из oracle_ceiling для этого не годится — он
    # посчитан по mean-метке, а ранжирование здесь против min (docs/05, 5.6).
    ceiling = {}
    if os.path.exists("results/analysis_relabel.json"):
        with open("results/analysis_relabel.json", encoding="utf-8") as fp:
            relabel = json.load(fp)
        for gen, variants_res in relabel.items():
            ref = variants_res.get(f"soft/{SCRMSD_AGG}") or {}
            ceiling[gen] = ref.get("median_oracle_lift_vs_B")

    print("Загрузка моделей...")
    variants = []  # (имя, модель, readout)
    soft = build_model("checkpoints/soft_model.pth", device)
    variants.append(("soft", soft, "fold"))
    if os.path.exists("checkpoints/perres_model.pth"):
        perres = build_model("checkpoints/perres_model.pth", device, per_residue=True)
        variants.append(("perres", perres, "lddt"))
    if os.path.exists("checkpoints/joint_model.pth"):
        joint = build_model("checkpoints/joint_model.pth", device, per_residue=True)
        variants.append(("joint/fold", joint, "fold"))
        variants.append(("joint/lddt", joint, "lddt"))

    results = {}
    for gen in GENERATOR_DIRS:
        # структуры и ground truth парсятся один раз на генератор
        records, _ = parse_pdb_files(GENERATOR_DIRS[gen])
        gt = scaffold_table(EVAL_SOURCES[gen])
        results[gen] = {"oracle_lift_ceiling": ceiling.get(gen)}
        for name, model, readout in variants:
            results[gen][name] = summarise(score_designability(model, records, device, readout), gt)

    hold = set(HOLDOUT_GENERATORS)
    print()
    print(f"{'генератор':16} {'режим':11} {'Spearman':>9} {'lift':>7}    потолок оракула")
    print("-" * 76)
    for gen, r in results.items():
        tag = "  ХОЛДАУТ" if gen in hold else ""
        print(f"{gen}{tag}")
        for name, _, _ in variants:
            v = r[name]
            sp, lf = v["median_spearman"], v["median_lift"]
            sp_s = f"{sp:>9.3f}" if sp is not None else f"{'—':>9}"
            lf_s = f"{lf:>6.2f}x" if lf is not None else f"{'—':>7}"
            ceil = r["oracle_lift_ceiling"]
            ceil_s = f"потолок {ceil:.2f}x" if ceil is not None else "потолок неизвестен"
            print(f"{'':16} {name:11} {sp_s} {lf_s}    ({ceil_s})")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}")


if __name__ == "__main__":
    main()
