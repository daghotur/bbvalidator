"""
analysis/oracle_ceiling.py
--------------------------
Измеряет потолок самого оракула self-consistency: сколько внутримотивного
ранжирующего сигнала вообще существует в метке scRMSD, независимо от модели.

scRMSD скаффолда — среднее RMSD восьми ESMFold-рефолдов восьми
ProteinMPNN-последовательностей. Это оценка по выборке, а не истина: у неё
есть собственный шум. Если шум сопоставим с межкандидатным разбросом внутри
мотива, то внутримотивное ранжирование ограничено оракулом, а не моделью.

Метод — split-half по последовательностям (не по структурам):
  * 8 последовательностей случайно делятся на 4 + 4;
  * scRMSD_A и scRMSD_B — независимые оценки одной и той же величины;
  * внутримотивный Spearman(A, B) = насколько оракул воспроизводит собственное
    ранжирование. Поправка Спирмена-Брауна проецирует с 4 на 8 последовательностей;
  * lift, посчитанный «ранжируем по A, проверяем по B», — эмпирический потолок
    в тех же единицах, что и lift модели (docs/08). Это НИЖНЯЯ оценка потолка:
    половинка шумнее, чем идеальная модель, знающая истинное среднее.

**Здесь и только здесь метка агрегируется средним, а не min.** Поправка
Спирмена-Брауна выведена для среднеподобных агрегатов и к min неприменима
(docs/06), поэтому надёжность R измерима только для mean-метки. Следствие:
делить Spearman модели (она меряется против min-метки) на sqrt(R) отсюда —
значит сравнивать числа, посчитанные против разных истин. Такое отношение
раньше печаталось как «доля забранного сигнала»; оно убрано. Честное
сравнение модели с оракулом, где обе стороны меряются одной меткой на общей
половине B, делает analysis/relabel.py — числа берутся оттуда.

Запуск:  python -m analysis.oracle_ceiling
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.motifbench import DESIGNABLE_MAX_SCRMSD, EVAL_SOURCES, per_sequence_rmsd
from common.ranking import (
    MIN_SEQUENCES_FOR_SPLIT,
    N_SPLITS,
    RNG_SEED,
    SATURATED_HIGH,
    SATURATED_LOW,
    lift_at_top,
    split_half,
)

OUT_JSON = "results/analysis_oracle_ceiling.json"


def spearman_brown(r: float, factor: float = 2.0) -> float:
    """Проекция надёжности с половины (4 seq) на полную длину (8 seq)."""
    if np.isnan(r) or r <= -1:
        return np.nan
    return factor * r / (1.0 + (factor - 1.0) * r)


def analyse(generator: str, root: str, rng: np.random.Generator) -> dict:
    per_seq = per_sequence_rmsd(root, min_sequences=MIN_SEQUENCES_FOR_SPLIT)
    by_motif = {}
    for (motif, sample), rmsd in per_seq.items():
        by_motif.setdefault(motif, []).append(rmsd)

    rows = []
    for motif, samples in by_motif.items():
        n = len(samples)
        if n < 10:
            continue
        full_mean = np.array([s.mean() for s in samples])
        base_rate = float((full_mean < DESIGNABLE_MAX_SCRMSD).mean())

        sp_halves, lifts, flip_rates = [], [], []
        for _ in range(N_SPLITS):
            a, b = split_half(samples, np.mean, rng)
            if len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
                r, _ = spearmanr(a, b)
                sp_halves.append(r)
            lifts.append(lift_at_top(a, b < DESIGNABLE_MAX_SCRMSD))
            # доля образцов, у которых бинарная метка дизайнуемости переворачивается
            flip_rates.append(float(
                ((a < DESIGNABLE_MAX_SCRMSD) != (b < DESIGNABLE_MAX_SCRMSD)).mean()
            ))

        r_half = float(np.nanmean(sp_halves)) if sp_halves else np.nan
        rows.append({
            "motif": motif,
            "n": n,
            "base_rate": base_rate,
            "oracle_spearman_half": r_half,
            "oracle_spearman_full": spearman_brown(r_half),
            "oracle_lift_ceiling": float(np.nanmean(lifts)),
            "label_flip_rate": float(np.nanmean(flip_rates)),
        })

    ms = pd.DataFrame(rows)
    kept = ms[(ms["base_rate"] <= SATURATED_HIGH) & (ms["base_rate"] >= SATURATED_LOW)]

    out_csv = f"results/oracle_ceiling_{generator.replace('/', '_')}.csv"
    ms.to_csv(out_csv, index=False)

    return {
        "generator": generator,
        "n_motifs": int(len(ms)),
        "n_kept": int(len(kept)),
        "median_oracle_spearman_full_kept": float(kept["oracle_spearman_full"].median()) if len(kept) else None,
        "median_oracle_lift_ceiling_kept": float(kept["oracle_lift_ceiling"].median()) if len(kept) else None,
        "median_label_flip_rate_kept": float(kept["label_flip_rate"].median()) if len(kept) else None,
        "csv": out_csv,
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    model = {}
    if os.path.exists("results/analysis_motif_bias.json"):
        with open("results/analysis_motif_bias.json", encoding="utf-8") as fp:
            model = json.load(fp)

    results = {}
    for gen, root in EVAL_SOURCES.items():
        res = analyse(gen, root, rng)
        m = model.get(gen, {})
        # Числа модели даются справочно и НЕ делятся на потолок отсюда: они
        # посчитаны против min-метки, а R — против mean (см. докстроку модуля).
        res["model_median_spearman_kept_min_label"] = m.get("median_spearman_kept")
        res["model_median_lift_kept_min_label"] = m.get("median_lift_kept")
        results[gen] = res

        print(f"\n=== {gen} ===")
        print(f"  мотивов {res['n_motifs']}, немасыщенных {res['n_kept']}")
        if res["n_kept"] == 0:
            print("  ненасыщенных мотивов нет — потолок оракула не определён")
            continue
        print(f"  ОРАКУЛ: надёжность R метки (8 seq, Spearman-Brown) = "
              f"{res['median_oracle_spearman_full_kept']:.3f}"
              f" → потолок Spearman sqrt(R) = {np.sqrt(res['median_oracle_spearman_full_kept']):.3f}")
        print(f"  ОРАКУЛ: потолок lift (ранж. по половине, проверка по другой) = "
              f"{res['median_oracle_lift_ceiling_kept']:.2f}x")
        print(f"  ОРАКУЛ: доля переворачивающихся бинарных меток = "
              f"{res['median_label_flip_rate_kept']:.3f}")
        print(f"  МОДЕЛЬ (справочно, метка min): within-motif Spearman = "
              f"{res['model_median_spearman_kept_min_label']}, "
              f"lift = {res['model_median_lift_kept_min_label']}")
        print("  → доля забранного сигнала здесь не считается: R измерена по "
              "mean-метке. Сравнение модели с потолком — python -m analysis.relabel")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}")


if __name__ == "__main__":
    main()
