"""
analysis/label_choice.py
------------------------
Выбор агрегатора scRMSD по 8 последовательностям: mean против min.

Проблема. docs/07 определяет метрику MotifBench как
    scRMSD = min_k RMSD(X_design, X_pred(s_k)),
а analysis/scrmsd.py считает rmsd.mean(). Все обученные модели и все числа
в docs/05 и docs/08 получены против среднего, то есть против величины, которой
нет ни в спецификации бенчмарка, ни в постановке скрининга.

Почему это не косметика: per-sequence RMSD резко бимодален (либо ~1 A, либо
>6 A), поэтому среднее по 8 — это в основном ДОЛЯ УСПЕШНЫХ последовательностей,
а min — «свернулся ли остов хоть раз». Для скрининга (какие остова пускать
в дорогой пайплайн) решение принимается по второму.

Цена перехода: min — порядковая статистика, она шумнее среднего. Поэтому здесь
меряется не только смещение вердикта, но и надёжность обеих меток при РАВНОМ
бюджете (split-half 4 + 4) — чтобы отличить реальное рассогласование модели
от возросшего шума метки.

Замечание о методе: поправка Спирмена-Брауна выведена для среднеподобных
агрегатов и к min неприменима, поэтому надёжность приводится сырой, на
половинном бюджете, одинаково для обоих агрегаторов.

Запуск:  python -m analysis.label_choice
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.motifbench import (
    AGGREGATORS,
    DESIGNABLE_MAX_SCRMSD,
    EVAL_SOURCES,
    per_sequence_rmsd,
)
from common.ranking import (
    MIN_SAMPLES_PER_MOTIF,
    MIN_SEQUENCES_FOR_SPLIT,
    N_SPLITS,
    RNG_SEED,
    SATURATED_HIGH,
    SATURATED_LOW,
    lift_at_top,
    split_half,
)

OUT_JSON = "results/analysis_label_choice.json"


def analyse_motif(samples: list[np.ndarray], agg, rng) -> dict:
    """Надёжность и потолок агрегатора agg на одном мотиве."""
    full = np.array([agg(s) for s in samples])
    base_rate = float((full < DESIGNABLE_MAX_SCRMSD).mean())

    rel, ceil, flip = [], [], []
    for _ in range(N_SPLITS):
        a, b = split_half(samples, agg, rng)
        if len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
            rel.append(spearmanr(a, b)[0])
        ceil.append(lift_at_top(a, b < DESIGNABLE_MAX_SCRMSD))
        flip.append(float(
            ((a < DESIGNABLE_MAX_SCRMSD) != (b < DESIGNABLE_MAX_SCRMSD)).mean()
        ))

    return {
        "base_rate": base_rate,
        "reliability_half": float(np.nanmean(rel)) if rel else np.nan,
        "lift_ceiling": float(np.nanmean(ceil)),
        "flip_rate": float(np.nanmean(flip)),
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    results = {}

    for gen, root in EVAL_SOURCES.items():
        per_seq = per_sequence_rmsd(root, min_sequences=MIN_SEQUENCES_FOR_SPLIT)
        by_motif: dict[str, list[np.ndarray]] = {}
        for (motif, sample), rmsd in per_seq.items():
            by_motif.setdefault(motif, []).append(rmsd)

        res = {}
        for name, agg in AGGREGATORS.items():
            rows = []
            for motif, samples in by_motif.items():
                if len(samples) < MIN_SAMPLES_PER_MOTIF:
                    continue
                rows.append({
                    "motif": motif,
                    "n": len(samples),
                    **analyse_motif(samples, agg, rng),
                })
            ms = pd.DataFrame(rows)
            kept = ms[(ms["base_rate"] <= SATURATED_HIGH)
                      & (ms["base_rate"] >= SATURATED_LOW)]
            res[name] = {
                "n_motifs": int(len(ms)),
                "n_kept": int(len(kept)),
                "pooled_base_rate": float(ms["base_rate"].mean()) if len(ms) else None,
                "median_reliability_half": float(kept["reliability_half"].median()) if len(kept) else None,
                "median_lift_ceiling": float(kept["lift_ceiling"].median()) if len(kept) else None,
                "median_flip_rate": float(kept["flip_rate"].median()) if len(kept) else None,
            }

        # насколько расходится бинарный вердикт дизайнируемости
        disagree = np.mean([
            (np.mean(s) < DESIGNABLE_MAX_SCRMSD) != (np.min(s) < DESIGNABLE_MAX_SCRMSD)
            for s in per_seq.values()
        ])
        res["verdict_disagreement"] = float(disagree)
        results[gen] = res

    print(f"{'генератор':16} {'агрегатор':10} {'дизайн.':>8} {'надёжн.':>8} "
          f"{'потолок lift':>13} {'флип':>7} {'мотивов':>8}")
    print("-" * 78)
    for gen, res in results.items():
        for name in AGGREGATORS:
            r = res[name]
            print(f"{gen if name == 'mean' else '':16} {name:10} "
                  f"{r['pooled_base_rate']:>7.1%} {r['median_reliability_half']:>8.3f} "
                  f"{r['median_lift_ceiling']:>12.2f}x {r['median_flip_rate']:>7.3f} "
                  f"{r['n_kept']:>4d}/{r['n_motifs']:<3d}")
        print(f"{'':16} {'расхождение вердикта mean/min: '}{res['verdict_disagreement']:.1%}")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}")


if __name__ == "__main__":
    main()
