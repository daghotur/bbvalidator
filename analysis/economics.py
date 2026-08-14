"""
analysis/economics.py
---------------------
Перевод ранжирующего качества в то, ради чего строился фильтр: сколько дорогих
проверок экономится и сколько хороших кандидатов при этом теряется.

Постановка. Дизайнер сгенерировал N остовов под конкретный мотив и проверяет их
дорогим пайплайном (ProteinMPNN x8 + ESMFold x8), пока не получит рабочий. Без
фильтра порядок проверки случаен, с фильтром — по возрастанию предсказанного
scRMSD. Вопрос: на сколько раньше находится первый успех.

Почему именно «первый успех», а не precision@K: это ровно то решение, которое
принимается на практике — проверять кандидатов, пока один не сработает. Метрика
не зависит от произвольного выбора глубины K и напрямую переводится в часы.

Считается два режима, они отвечают на разные вопросы:
  * ПО-МОТИВНО   — у дизайнера одна цель, кандидаты только под неё (основной);
  * ПУЛОВО       — фиксированный бюджет на много целей сразу. Здесь «вычерпывание
                   лёгких мотивов», которое портит пуловую метрику качества
                   модели (docs/08), становится рациональным поведением, а не
                   артефактом: цель — максимум успехов на единицу бюджета.

Оговорка о цене: стоимость дорогой проверки взята оценкой (8 рефолдов ESMFold),
поэтому часы приводятся с явным диапазоном. Отношение цен, а не абсолютные
часы — то, что здесь надёжно.

Запуск:  python -m analysis.economics
"""

import json

import numpy as np
import pandas as pd
import torch

from common.motifbench import (
    AGGREGATORS,
    DESIGNABLE_MAX_SCRMSD,
    EVAL_SOURCES,
    GENERATOR_DIRS,
    HOLDOUT_GENERATORS,
    SCRMSD_AGG,
    per_sequence_rmsd,
)
from common.ranking import MIN_SEQUENCES_FOR_SPLIT
from common.scoring import score_designability
from common.structures import parse_pdb_files
from inference import build_model

REPEATS = 3            # усреднение предсказаний против bf16-джиттера
BUDGETS = (1, 3, 5, 10, 20)
FILTER_RATE = 230.0    # структур/с, полный прогон filter_designability (docs/05 5.5)
VERIFY_SECONDS = (10.0, 20.0, 60.0)   # 8 рефолдов ESMFold: нижняя, средняя, верхняя
OUT_JSON = "results/analysis_economics.json"
OUT_CSV = "results/economics_per_motif.csv"


def hypergeom_hit(n: int, d: int, k: int) -> float:
    """P(хотя бы один успех) при случайном выборе k из n, где d успешных."""
    if d == 0:
        return 0.0
    if k >= n - d:
        return 1.0
    # P(ни одного) = C(n-d, k) / C(n, k), считаем произведением, без факториалов
    p_none = 1.0
    for i in range(k):
        p_none *= (n - d - i) / (n - i)
    return 1.0 - p_none


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agg = AGGREGATORS[SCRMSD_AGG]
    model = build_model("checkpoints/joint_model.pth", device, per_residue=True)

    rows = []
    pooled = {}
    for gen in GENERATOR_DIRS:
        records, _ = parse_pdb_files(GENERATOR_DIRS[gen])
        per_seq = per_sequence_rmsd(EVAL_SOURCES[gen], min_sequences=MIN_SEQUENCES_FOR_SPLIT)

        preds = [score_designability(model, records, device) for _ in range(REPEATS)]
        mean_pred = np.mean([p["pred_scrmsd"].to_numpy() for p in preds], axis=0)
        base = preds[0].assign(pred=mean_pred)

        base["key"] = list(zip(base["motif"], base["sample"]))
        base = base[base["key"].isin(per_seq)].copy()
        base["truth"] = [agg(per_seq[k]) for k in base["key"]]
        base["design"] = base["truth"] < DESIGNABLE_MAX_SCRMSD

        # ---- пуловый режим
        s = base.sort_values("pred")
        pooled[gen] = {
            "n": int(len(base)),
            "base_rate": float(base["design"].mean()),
            "first_hit_model": int(np.argmax(s["design"].to_numpy()) + 1)
            if s["design"].any() else None,
            "first_hit_random": float((len(base) + 1) / (base["design"].sum() + 1)),
        }

        # ---- по-мотивный режим
        for motif, g in base.groupby("motif"):
            n, d = len(g), int(g["design"].sum())
            gs = g.sort_values("pred")
            hit = gs["design"].to_numpy()
            rec = {
                "generator": gen, "motif": motif, "n": n, "n_design": d,
                "base_rate": d / n,
                "first_model": int(np.argmax(hit) + 1) if d else None,
                "first_random": (n + 1) / (d + 1) if d else None,
            }
            for k in BUDGETS:
                rec[f"model_k{k}"] = bool(hit[:k].any())
                rec[f"random_k{k}"] = hypergeom_hit(n, d, k)
            rows.append(rec)

    ms = pd.DataFrame(rows)
    ms.to_csv(OUT_CSV, index=False)
    solvable = ms[ms["n_design"] > 0].copy()
    solvable["speedup"] = solvable["first_random"] / solvable["first_model"]

    print(f"\nМотивов всего {len(ms)}, из них хотя бы с одним дизайнируемым "
          f"кандидатом {len(solvable)} ({len(solvable) / len(ms):.0%})")
    print(f"На {len(ms) - len(solvable)} мотивах дизайнируемых нет вообще — "
          f"там никакой фильтр не поможет.\n")

    hold = set(HOLDOUT_GENERATORS)
    print("ПО-МОТИВНО: сколько дорогих проверок до первого успеха")
    print(f"{'генератор':16} {'случайно':>9} {'с фильтром':>11} {'выигрыш':>9} "
          f"{'хуже случая':>12}")
    print("-" * 62)
    for gen, g in solvable.groupby("generator"):
        tag = "*" if gen in hold else " "
        print(f"{gen + tag:16} {g['first_random'].median():>9.1f} "
              f"{g['first_model'].median():>11.1f} "
              f"{g['speedup'].median():>8.2f}x {(g['speedup'] < 1).mean():>11.0%}")
    print(f"{'ВСЕ':16} {solvable['first_random'].median():>9.1f} "
          f"{solvable['first_model'].median():>11.1f} "
          f"{solvable['speedup'].median():>8.2f}x "
          f"{(solvable['speedup'] < 1).mean():>11.0%}      (* — холдаут)")

    print("\nДОЛЯ ЦЕЛЕЙ, ЗАКРЫТЫХ ЗА K ПРОВЕРОК (по всем мотивам, включая безнадёжные)")
    print(f"{'K':>3} {'случайно':>10} {'с фильтром':>12} {'прирост':>9}")
    print("-" * 38)
    budget_rows = {}
    for k in BUDGETS:
        r = ms[f"random_k{k}"].mean()
        m = ms[f"model_k{k}"].mean()
        budget_rows[k] = {"random": float(r), "model": float(m)}
        print(f"{k:>3} {r:>9.1%} {m:>11.1%} {(m - r) * 100:>+6.1f} п.п.")

    # Главный разрез: выигрыш фильтра — функция сложности цели. На лёгких
    # мишенях случайный выбор и так срабатывает с первой-второй попытки.
    print("\nВЫИГРЫШ В ЗАВИСИМОСТИ ОТ СЛОЖНОСТИ ЦЕЛИ")
    print(f"{'доля дизайнируемых':>20} {'мотивов':>8} {'случайно':>9} "
          f"{'с фильтром':>11} {'выигрыш':>9}")
    print("-" * 62)
    bins = [(0.0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    by_difficulty = {}
    for lo, hi in bins:
        sel = solvable[(solvable["base_rate"] >= lo) & (solvable["base_rate"] < hi)]
        if sel.empty:
            continue
        by_difficulty[f"{lo:.2f}-{hi:.2f}"] = {
            "n_motifs": int(len(sel)),
            "median_first_random": float(sel["first_random"].median()),
            "median_first_model": float(sel["first_model"].median()),
            "median_speedup": float(sel["speedup"].median()),
        }
        print(f"{f'{lo:.0%}-{hi:.0%}':>20} {len(sel):>8} "
              f"{sel['first_random'].median():>9.1f} "
              f"{sel['first_model'].median():>11.1f} "
              f"{sel['speedup'].median():>8.2f}x")

    print("\nЦЕНА ОДНОЙ ДОРОГОЙ ПРОВЕРКИ ПРОТИВ ПРОГОНА ФИЛЬТРА")
    for v in VERIFY_SECONDS:
        print(f"  проверка {v:>4.0f} с → одна проверка = {v * FILTER_RATE:>6.0f} "
              f"прогонов фильтра")

    hard = solvable[solvable["base_rate"] < 0.25]
    scenarios = {}
    for name, sub in (("все цели", solvable), ("только трудные (<25%)", hard)):
        med_r = sub["first_random"].median()
        med_m = sub["first_model"].median()
        print(f"\nСЦЕНАРИЙ ({name}): 100 целей, по 100 кандидатов, "
              f"нужен один рабочий остов")
        print(f"{'проверка, с':>12} {'без фильтра':>14} {'с фильтром':>14} {'экономия':>12}")
        print("-" * 56)
        scenarios[name] = {}
        for v in VERIFY_SECONDS:
            h_no = 100 * med_r * v / 3600
            h_yes = (100 * med_m * v + 100 * 100 / FILTER_RATE) / 3600
            scenarios[name][v] = {"hours_without": h_no, "hours_with": h_yes}
            print(f"{v:>12.0f} {h_no:>13.1f}ч {h_yes:>13.1f}ч "
                  f"{1 - h_yes / h_no:>11.0%}")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump({
            "per_motif": {
                "n_motifs": int(len(ms)),
                "n_solvable": int(len(solvable)),
                "median_first_random": float(solvable["first_random"].median()),
                "median_first_model": float(solvable["first_model"].median()),
                "median_speedup": float(solvable["speedup"].median()),
                "frac_worse_than_random": float((solvable["speedup"] < 1).mean()),
            },
            "budgets": budget_rows,
            "by_difficulty": by_difficulty,
            "pooled": pooled,
            "scenarios": {k: {str(kk): vv for kk, vv in v.items()}
                          for k, v in scenarios.items()},
        }, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}, {OUT_CSV}")


if __name__ == "__main__":
    main()
