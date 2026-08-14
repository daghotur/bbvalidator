"""
analysis/second_oracle.py
-------------------------
Проверка на артефакты оракула: учится ли модель дизайнируемости — или причудам
конкретного предсказателя структуры.

Вся супервизия проекта построена на self-consistency через ESMFold. Известно
(docs/07), что результат сильно зависит от рефолдера: на MotifBench RFdiffusion
набирает 28.1 с ESMFold и 22.5 с AF2. Если наш скор коррелирует только с
ESMFold-истиной, мы построили детектор вкусов ESMFold, а не скорер
дизайнируемости.

MotifBench свернул ТЕ ЖЕ восемь последовательностей обоими предсказателями
(af2_eval_results.csv рядом с esm_eval_results.csv) — второй оракул считать не
нужно, он уже на диске. Доступен для RFdiffusion (обучающий генератор) и GPDL
(холдаут).

Решающее сравнение — три внутримотивных корреляции:
  * Spearman(ESM, AF2)    — сколько общего сигнала вообще есть у двух оракулов;
  * Spearman(модель, ESM) — то, что мы отчитываем (модель обучена на ESM);
  * Spearman(модель, AF2) — перенос на невиданный оракул.

Читается так: если Spearman(модель, AF2) ≈ Spearman(ESM, AF2), то модель
предсказывает AF2 не хуже, чем сам ESMFold, — значит она забрала общую физику,
а не идиосинкразию ESMFold. Если сильно меньше — забрала идиосинкразию.

Запуск:  python -m analysis.second_oracle
"""

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common.motifbench import (
    AGGREGATORS,
    DESIGNABLE_MAX_SCRMSD,
    EVAL_SOURCES,
    GENERATOR_DIRS,
    HOLDOUT_GENERATORS,
    SCRMSD_AGG,
    per_sequence_rmsd,
)
from common.ranking import (
    MIN_SAMPLES_PER_MOTIF,
    MIN_SEQUENCES_FOR_SPLIT,
    N_SPLITS,
    RNG_SEED,
    SATURATED_HIGH,
    SATURATED_LOW,
    precision_at_top,
    split_half,
)
from common.scoring import score_designability, score_lookup
from common.structures import parse_pdb_files
from inference import build_model

OUT_JSON = "results/analysis_second_oracle.json"


def oracle_ceiling(samples: list[np.ndarray], agg, rng) -> float:
    """Потолок lift на данной метке: ранжируем половиной A, проверяем половиной B."""
    lifts = []
    for _ in range(N_SPLITS):
        a, b = split_half(samples, agg, rng)
        design_b = b < DESIGNABLE_MAX_SCRMSD
        if design_b.all() or not design_b.any():
            continue
        lifts.append(precision_at_top(a, design_b) / design_b.mean())
    return float(np.mean(lifts)) if lifts else np.nan


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agg = AGGREGATORS[SCRMSD_AGG]
    model = build_model("checkpoints/joint_model.pth", device, per_residue=True)

    results = {}
    for gen, root in EVAL_SOURCES.items():
        af2 = per_sequence_rmsd(root, "AF2", MIN_SEQUENCES_FOR_SPLIT)
        if not af2:
            continue
        esm = per_sequence_rmsd(root, "ESMFold", MIN_SEQUENCES_FOR_SPLIT)
        common = sorted(set(esm) & set(af2))

        records, _ = parse_pdb_files(GENERATOR_DIRS[gen])
        lookup = score_lookup(score_designability(model, records, device, "fold"))

        by_motif: dict[str, list] = {}
        for key in common:
            p = lookup.get(key)
            if p is not None:
                by_motif.setdefault(key[0], []).append((p, esm[key], af2[key]))

        rng = np.random.default_rng(RNG_SEED)
        rows = []
        for motif, items in by_motif.items():
            if len(items) < MIN_SAMPLES_PER_MOTIF:
                continue
            p = np.array([x[0] for x in items])
            e_seq = [x[1] for x in items]
            a_seq = [x[2] for x in items]
            e = np.array([agg(s) for s in e_seq])
            a = np.array([agg(s) for s in a_seq])
            d_e, d_a = e < DESIGNABLE_MAX_SCRMSD, a < DESIGNABLE_MAX_SCRMSD

            row = {
                "motif": motif,
                "n": len(items),
                "base_esm": float(d_e.mean()),
                "base_af2": float(d_a.mean()),
                "verdict_disagree": float((d_e != d_a).mean()),
                "sp_esm_af2": spearmanr(e, a)[0],
                "sp_model_esm": spearmanr(p, e)[0],
                "sp_model_af2": spearmanr(p, a)[0],
            }
            # lift и потолок считаются только на ненасыщенных по данному оракулу
            for tag, d, seqs in (("esm", d_e, e_seq), ("af2", d_a, a_seq)):
                if SATURATED_LOW <= d.mean() <= SATURATED_HIGH:
                    row[f"lift_{tag}"] = precision_at_top(p, d) / d.mean()
                    row[f"ceiling_{tag}"] = oracle_ceiling(seqs, agg, rng)
            rows.append(row)

        ms = pd.DataFrame(rows)
        out_csv = f"results/second_oracle_{gen.replace('/', '_')}.csv"
        ms.to_csv(out_csv, index=False)

        res = {"n_motifs": int(len(ms)), "n_samples": int(ms["n"].sum()),
               "csv": out_csv}
        # Парный по мотивам разрыв «своего» и «чужого» оракула: если модель
        # выучила идиосинкразию ESMFold, разрыв должен быть устойчиво > 0.
        gap = ms["sp_model_esm"] - ms["sp_model_af2"]
        boot = np.array([
            np.median(rng.choice(gap.to_numpy(), size=len(gap), replace=True))
            for _ in range(2000)
        ])
        res["gap_median"] = float(gap.median())
        res["gap_ci90"] = [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))]
        res["gap_frac_positive"] = float((gap > 0).mean())
        for c in ["base_esm", "base_af2", "verdict_disagree",
                  "sp_esm_af2", "sp_model_esm", "sp_model_af2"]:
            res[c] = float(ms[c].median())
        for tag in ("esm", "af2"):
            sub = ms.dropna(subset=[f"lift_{tag}"]) if f"lift_{tag}" in ms else ms.iloc[:0]
            res[f"n_kept_{tag}"] = int(len(sub))
            res[f"lift_{tag}"] = float(sub[f"lift_{tag}"].median()) if len(sub) else None
            res[f"ceiling_{tag}"] = float(sub[f"ceiling_{tag}"].median()) if len(sub) else None
            res[f"frac_ceiling_{tag}"] = float(
                (sub[f"lift_{tag}"] / sub[f"ceiling_{tag}"]).median()
            ) if len(sub) else None
        # ключевой показатель: предсказываем ли мы AF2 не хуже, чем это делает ESMFold
        res["transfer_ratio"] = res["sp_model_af2"] / res["sp_esm_af2"]
        results[gen] = res

    hold = set(HOLDOUT_GENERATORS)
    for gen, r in results.items():
        tag = "  (ХОЛДАУТ)" if gen in hold else "  (обучающий)"
        print(f"\n=== {gen}{tag} — {r['n_samples']} структур, {r['n_motifs']} мотивов ===")
        print(f"  базовая доля дизайнируемых: ESMFold {r['base_esm']:.1%}, AF2 {r['base_af2']:.1%}")
        print(f"  вердикт расходится между оракулами: {r['verdict_disagree']:.1%}")
        print(f"  СОГЛАСИЕ ОРАКУЛОВ    Spearman(ESM, AF2)   = {r['sp_esm_af2']:.3f}")
        print(f"  МОДЕЛЬ на своём      Spearman(модель, ESM) = {r['sp_model_esm']:.3f}")
        print(f"  МОДЕЛЬ на чужом      Spearman(модель, AF2) = {r['sp_model_af2']:.3f}")
        print(f"  → перенос: {r['transfer_ratio']:.0%} от того, что даёт сам ESMFold")
        print(f"  разрыв свой−чужой по мотивам: медиана {r['gap_median']:+.3f}, "
              f"90% CI [{r['gap_ci90'][0]:+.3f}, {r['gap_ci90'][1]:+.3f}], "
              f"положителен у {r['gap_frac_positive']:.0%} мотивов")
        for tag_, name in (("esm", "ESMFold"), ("af2", "AF2")):
            if r[f"lift_{tag_}"] is not None:
                print(f"  lift против {name:8} {r[f'lift_{tag_}']:.2f}x "
                      f"(потолок {r[f'ceiling_{tag_}']:.2f}x, "
                      f"{r[f'frac_ceiling_{tag_}']:.0%}), мотивов {r[f'n_kept_{tag_}']}")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}")


if __name__ == "__main__":
    main()
