"""
analysis_perresidue.py
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

Опорная точка — потолок оракула (analysis_oracle_ceiling.py): сколько
ранжирующего сигнала вообще есть в метке.

Запуск:  python analysis_perresidue.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from analysis_logits_ranking import EVAL_SOURCES, GENERATOR_DIRS, parse_pdb_files
from analysis_motif_bias import SATURATED_HIGH, SATURATED_LOW, per_motif_stats
from analysis_scrmsd import parse_motifbench_eval
from inference import _autocast_ctx, build_model, center_coords
from train_soft import HOLDOUT_GENERATORS

BATCH = 32
OUT_JSON = "analysis_perresidue.json"


@torch.no_grad()
def score(model, records, device, readout: str) -> pd.DataFrame:
    """Скор в единицах «меньше = дизайнируемее», чтобы все режимы сравнивались
    одной формулой lift/Spearman."""
    order = sorted(range(len(records)), key=lambda i: len(records[i]["coords"]))
    rows = [None] * len(records)
    for start in range(0, len(order), BATCH):
        idxs = order[start : start + BATCH]
        Lmax = max(len(records[i]["coords"]) for i in idxs)
        B = len(idxs)
        coords = np.zeros((B, Lmax, 3, 3), dtype=np.float32)
        mask = np.zeros((B, Lmax), dtype=np.bool_)
        for b, i in enumerate(idxs):
            L = len(records[i]["coords"])
            coords[b, :L] = center_coords(records[i]["coords"])
            mask[b, :L] = True
        c = torch.from_numpy(coords).to(device)
        m = torch.from_numpy(mask).to(device)
        with _autocast_ctx(device):
            out = model(c, m)
        if readout == "lddt":
            lddt = torch.sigmoid(out["lddt_logit"].float())
            mf = m.float()
            # больше lDDT = лучше; инвертируем знак, чтобы «меньше = лучше»
            s = -((lddt * mf).sum(-1) / mf.sum(-1).clamp(min=1))
        else:
            s = torch.clamp(torch.expm1(out["fold_logit"].float()), min=0.0)
        s = s.cpu().numpy()
        for b, i in enumerate(idxs):
            rows[i] = {
                "sample": records[i]["sample"].removesuffix(".pdb"),
                "motif": records[i]["motif"],
                "pred_scrmsd": float(s[b]),
            }
    return pd.DataFrame(rows)


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

    ceiling = {}
    if os.path.exists("analysis_oracle_ceiling.json"):
        with open("analysis_oracle_ceiling.json", encoding="utf-8") as fp:
            ceiling = json.load(fp)

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
        gt = parse_motifbench_eval(EVAL_SOURCES[gen])
        c = ceiling.get(gen, {})
        results[gen] = {
            "oracle_max_spearman": c.get("max_attainable_spearman"),
            "oracle_lift_ceiling": c.get("median_oracle_lift_ceiling_kept"),
        }
        for name, model, readout in variants:
            results[gen][name] = summarise(score(model, records, device, readout), gt)

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
            print(f"{'':16} {name:11} {sp_s} {lf_s}"
                  f"    (≤ {r['oracle_max_spearman']:.3f}, ≤ {r['oracle_lift_ceiling']:.2f}x)")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}")


if __name__ == "__main__":
    main()
