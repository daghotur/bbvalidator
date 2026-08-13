"""
analysis_oracle_ceiling.py
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

Сравнение с моделью: model_spearman / oracle_spearman — какую долю доступного
сигнала модель уже забирает.

Запуск:  python analysis_oracle_ceiling.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis_logits_ranking import EVAL_SOURCES

DESIGNABLE_MAX_SCRMSD = 2.0
N_SPLITS = 50
SATURATED_HIGH = 0.90
SATURATED_LOW = 0.10
RNG_SEED = 42
OUT_JSON = "analysis_oracle_ceiling.json"


def load_per_sequence_rmsd(root: str) -> dict:
    """(motif, sample) -> вектор из 8 per-sequence RMSD (не усреднённых)."""
    out = {}
    for csv_path in glob.glob(os.path.join(root, "**", "esm_eval_results.csv"), recursive=True):
        if "__MACOSX" in csv_path:
            continue
        parts = csv_path.split(os.sep)
        sample, motif = parts[-3], parts[-4]
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "rmsd" not in df.columns:
            continue
        rmsd = pd.to_numeric(df["rmsd"], errors="coerce").dropna().to_numpy()
        if len(rmsd) < 4:
            continue
        out[(motif, sample)] = rmsd
    return out


def spearman_brown(r: float, factor: float = 2.0) -> float:
    """Проекция надёжности с половины (4 seq) на полную длину (8 seq)."""
    if np.isnan(r) or r <= -1:
        return np.nan
    return factor * r / (1.0 + (factor - 1.0) * r)


def lift_from_ranking(rank_by: np.ndarray, truth: np.ndarray) -> float:
    """precision@top-10% (по rank_by) / base_rate, оба по truth."""
    n = len(truth)
    design = truth < DESIGNABLE_MAX_SCRMSD
    base = design.mean()
    if base == 0 or base == 1:
        return np.nan
    k = max(1, round(0.1 * n))
    top = np.argsort(rank_by)[:k]
    return design[top].mean() / base


def analyse(generator: str, root: str, rng: np.random.Generator) -> dict:
    per_seq = load_per_sequence_rmsd(root)
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
            a_means, b_means = [], []
            for s in samples:
                idx = rng.permutation(len(s))
                half = len(s) // 2
                a_means.append(s[idx[:half]].mean())
                b_means.append(s[idx[half:]].mean())
            a = np.array(a_means)
            b = np.array(b_means)
            if len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
                r, _ = spearmanr(a, b)
                sp_halves.append(r)
            lifts.append(lift_from_ranking(a, b))
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

    out_csv = f"oracle_ceiling_{generator.replace('/', '_')}.csv"
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
    if os.path.exists("analysis_motif_bias.json"):
        with open("analysis_motif_bias.json", encoding="utf-8") as fp:
            model = json.load(fp)

    results = {}
    for gen, root in EVAL_SOURCES.items():
        res = analyse(gen, root, rng)
        m = model.get(gen, {})
        res["model_median_spearman_kept"] = m.get("median_spearman_kept")
        res["model_median_lift_kept"] = m.get("median_lift_kept")
        # Аттенюация: corr(модель, наблюдаемое) = corr(модель, истина) * sqrt(R),
        # поэтому максимум достижимого Spearman против зашумлённой метки = sqrt(R).
        oc_sp = res["median_oracle_spearman_full_kept"]
        if oc_sp and res["model_median_spearman_kept"]:
            res["max_attainable_spearman"] = float(np.sqrt(oc_sp))
            res["signal_captured_frac"] = float(
                res["model_median_spearman_kept"] / np.sqrt(oc_sp)
            )
        results[gen] = res

        print(f"\n=== {gen} ===")
        print(f"  мотивов {res['n_motifs']}, немасыщенных {res['n_kept']}")
        print(f"  ОРАКУЛ: надёжность R метки (8 seq, Spearman-Brown) = "
              f"{res['median_oracle_spearman_full_kept']:.3f}"
              f" → потолок Spearman sqrt(R) = {np.sqrt(res['median_oracle_spearman_full_kept']):.3f}")
        print(f"  ОРАКУЛ: потолок lift (ранж. по половине, проверка по другой) = "
              f"{res['median_oracle_lift_ceiling_kept']:.2f}x")
        print(f"  ОРАКУЛ: доля переворачивающихся бинарных меток = "
              f"{res['median_label_flip_rate_kept']:.3f}")
        print(f"  МОДЕЛЬ: within-motif Spearman = {res['model_median_spearman_kept']}, "
              f"lift = {res['model_median_lift_kept']}")
        if "signal_captured_frac" in res:
            print(f"  → модель забирает {res['signal_captured_frac']:.0%} доступного сигнала")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}")


if __name__ == "__main__":
    main()
