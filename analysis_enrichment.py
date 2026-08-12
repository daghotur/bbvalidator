"""
analysis_enrichment.py
----------------------
Шаг 2 (ревизованный): фактор обогащения скрининга дизайнуемости.

Для 18k структур MotifBench считает, во сколько раз доля «дизайнуемых»
(scRMSD < 2 Å по self-consistency оракулу — вычислительный прокси, не
экспериментальная истина) в топ-q% по скореру выше базовой доли.

Обогащение считается POOLED и стратифицированно по генераторам: claim
делается по within-generator обогащению, pooled может отражать просто
распознавание генератора.

Скореры:
  ML (3 модели): сырой fold_logit, MC P(fold), rmsd-голова, p_steric;
  диагностика: MC-неопределённость;
  наивные бейзлайны: clash_pairs (геометрия, без весов), длина, random.
  Бейзлайны оцениваются в лучшей из двух направлений (даём им максимум шанса).

Запуск:  python analysis_enrichment.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from analysis_logits_ranking import EVAL_SOURCES, GENERATOR_DIRS, MODELS, parse_pdb_files
from eval_model import build_eval_model

DESIGNABLE_MAX_SCRMSD = 2.0
Q_GRID = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
DETAIL_CSV = "eval_results_generated_detail.csv"
OUT_CSV = "analysis_enrichment.csv"
OUT_JSON = "analysis_enrichment.json"


@torch.no_grad()
def geometry_clashes(records: list[dict], device: torch.device,
                     batch_size: int = 64) -> np.ndarray:
    """Сырые стерические конфликты (Cβ < 3.5 Å, |i-j| >= 3) — без весов модели."""
    from preprocess.biophys_frontend import BackboneGeometryExtractor

    extractor = BackboneGeometryExtractor().to(device)
    order = sorted(range(len(records)), key=lambda i: len(records[i]["coords"]))
    clashes = np.zeros(len(records), dtype=np.float32)

    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        Lmax = max(len(records[i]["coords"]) for i in idxs)
        B = len(idxs)
        coords = np.zeros((B, Lmax, 3, 3), dtype=np.float32)
        mask = np.zeros((B, Lmax), dtype=np.bool_)
        for b, i in enumerate(idxs):
            L = len(records[i]["coords"])
            coords[b, :L] = records[i]["coords"]
            mask[b, :L] = True
        geom = extractor(
            torch.from_numpy(coords).to(device), torch.from_numpy(mask).to(device)
        )
        per_struct = (geom["clash_count"] * torch.from_numpy(mask).to(device).float()).sum(-1) / 2.0
        for b, i in enumerate(idxs):
            clashes[i] = per_struct[b].item()
    return clashes


@torch.no_grad()
def score_deterministic(model, records: list[dict], device: torch.device,
                        batch_size: int = 32) -> pd.DataFrame:
    from inference import _autocast_ctx, center_coords

    order = sorted(range(len(records)), key=lambda i: len(records[i]["coords"]))
    rows = [None] * len(records)

    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        Lmax = max(len(records[i]["coords"]) for i in idxs)
        B = len(idxs)
        coords = np.zeros((B, Lmax, 3, 3), dtype=np.float32)
        mask = np.zeros((B, Lmax), dtype=np.bool_)
        for b, i in enumerate(idxs):
            L = len(records[i]["coords"])
            coords[b, :L] = center_coords(records[i]["coords"])
            mask[b, :L] = True

        with _autocast_ctx(device):
            preds = model(torch.from_numpy(coords).to(device),
                          torch.from_numpy(mask).to(device))

        logit = preds["fold_logit"].float().cpu().numpy()
        rmsd = preds["rmsd"].float().cpu().numpy()
        steric = torch.sigmoid(preds["steric"]).float().cpu().numpy()
        for b, i in enumerate(idxs):
            rows[i] = {
                "fold_logit": float(logit[b]),
                "rmsd_head": float(rmsd[b]),
                "p_steric": float(steric[b]),
            }
        if (start // batch_size) % 40 == 0:
            print(f"    скоринг: {start + B}/{len(records)}")

    return pd.DataFrame(rows)


def enrichment_curve(scores: np.ndarray, labels: np.ndarray, ascending: bool) -> dict:
    order = np.argsort(scores)
    if not ascending:
        order = order[::-1]
    base = labels.mean()
    n = len(labels)
    curve = {}
    for q in Q_GRID:
        k = max(1, int(q * n))
        top = labels[order[:k]]
        curve[str(q)] = {
            "enrichment": float(top.mean() / base) if base > 0 else float("nan"),
            "designable_in_bucket": int(top.sum()),
            "bucket_size": int(k),
        }
    return curve


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # 1. Парсинг один раз
    records_by_group = {}
    for group, root in GENERATOR_DIRS.items():
        records, skipped = parse_pdb_files(root)
        records_by_group[group] = records
        print(f"{group}: {len(records)} структур" + (f" (пропущено {skipped})" if skipped else ""))

    all_records, group_of = [], []
    for group, records in records_by_group.items():
        all_records.extend(records)
        group_of.extend([group] * len(records))
    print(f"Всего: {len(all_records)}")

    # 2. Геометрические конфликты (без весов)
    print("Геометрия: clash_pairs...")
    clash_pairs = geometry_clashes(all_records, device)

    # 3. Детерминированные головы трёх моделей
    scorer_cols = {}
    for model_name, ckpt in MODELS.items():
        print(f"Модель: {model_name}...")
        model, _ = build_eval_model(ckpt, device, pca_path="dataset/pca_components.pth")
        model.eval()
        df = score_deterministic(model, all_records, device)
        del model
        torch.cuda.empty_cache()
        for col in ["fold_logit", "rmsd_head", "p_steric"]:
            scorer_cols[f"{col}_{model_name}"] = df[col].values

    # 4. MC P(fold) / неопределённость / длина из существующих артефактов
    detail = pd.read_csv(DETAIL_CSV)
    detail["sample"] = detail["name"].str.replace(".pdb", "", regex=False)
    keys = [
        (r["motif"], r["sample"].removesuffix(".pdb"), g)
        for r, g in zip(all_records, group_of)
    ]
    for model_name in MODELS:
        m = detail["model"] == model_name
        sub = detail[m]
        sub_index = {
            (row.motif, row.sample, row.group): i
            for i, row in enumerate(sub.itertuples())
        }
        rows_idx = np.array([sub_index[k] for k in keys])
        scorer_cols[f"pfold_mc_{model_name}"] = sub["p_fold"].values[rows_idx]
        scorer_cols[f"uncertainty_{model_name}"] = sub["uncertainty"].values[rows_idx]

    length = np.array([len(r["coords"]) for r in all_records], dtype=np.float32)

    merged = pd.DataFrame(scorer_cols)
    merged["group"] = group_of
    merged["clash_pairs"] = clash_pairs
    merged["length"] = length

    # 5. Ground truth: scRMSD по генераторам (официальный формат MotifBench:
    #    среднее rmsd по дизайнуемым секвенциям — агрегация воспроизводит
    #    числа docs/05, табл. «Распределение scRMSD»)
    scrmsd = {}
    for group, root in EVAL_SOURCES.items():
        per_scaffold = {}
        for f in glob.glob(
            os.path.join(root, "*", "*", "*", "self_consistency", "esm_eval_results.csv")
        ):
            parts = f.split(os.sep)
            motif, sample = parts[-4], parts[-3]
            d = pd.read_csv(f)
            per_scaffold[(motif, sample)] = float(d["rmsd"].mean())
        scrmsd[group] = per_scaffold
        print(f"{group}: {len(per_scaffold)} скаффолдов со scRMSD")

    sc = np.array([
        scrmsd[g].get((r["motif"], r["sample"].removesuffix(".pdb")), np.nan)
        for r, g in zip(all_records, group_of)
    ])
    merged["sc_rmsd"] = sc
    valid = ~np.isnan(sc)
    merged = merged[valid].reset_index(drop=True)
    labels = (merged["sc_rmsd"] < DESIGNABLE_MAX_SCRMSD).values.astype(float)
    print(f"Со scRMSD: {len(merged)} | базовая доля дизайнуемых: {labels.mean():.3f}")

    # 6. Кривые обогащения: pooled + по генераторам
    # направление: False = выше скор → дизайнуемее; бейзлайны крутим в обе стороны
    directions = {}
    for k in scorer_cols:
        if k.startswith(("fold_logit", "pfold_mc")):
            directions[k] = [False]          # больше = лучше
        else:                                # rmsd_head, p_steric, uncertainty: меньше = лучше
            directions[k] = [True]
    directions["clash_pairs"] = [True, False]
    directions["length"] = [True, False]

    all_scorer_arrays = {**scorer_cols, "clash_pairs": clash_pairs, "length": length}

    results = {}
    for name, arr in all_scorer_arrays.items():
        best_curve_pooled, best_enr_pooled = None, -1.0
        for asc in directions[name]:
            curve = enrichment_curve(arr[valid], labels, asc)
            if curve[str(max(Q_GRID))]["enrichment"] > best_enr_pooled:
                best_enr_pooled = curve[str(max(Q_GRID))]["enrichment"]
                best_curve_pooled = (asc, curve)
        asc, curve = best_curve_pooled
        results[name] = {"direction_ascending": asc, "pooled": curve, "per_generator": {}}
        for group in GENERATOR_DIRS:
            gm = (merged["group"] == group).values
            if gm.sum() == 0:
                continue
            results[name]["per_generator"][group] = enrichment_curve(
                arr[valid][gm], labels[gm], asc
            )

    results["random"] = {
        "direction_ascending": None,
        "pooled": {str(q): {"enrichment": 1.0} for q in Q_GRID},
        "per_generator": {
            g: {str(q): {"enrichment": 1.0} for q in Q_GRID} for g in GENERATOR_DIRS
        },
    }

    merged.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    # 7. Таблица: обогащение при q=10%
    qk = "0.1"
    base_by_group = {
        g: float(labels[merged["group"].values == g].mean()) for g in GENERATOR_DIRS
    }
    print("\nБазовые доли дизайнуемых (scRMSD < 2 Å):")
    for g, b in base_by_group.items():
        n = int((merged["group"] == g).sum())
        print(f"  {g}: {b:.3f} (n={n})")

    print(f"\nОбогащение при топ-10% (бейзлайны — в лучшем направлении):")
    header = f"{'скорер':<24} | " + " | ".join(f"{g[:10]:>10}" for g in GENERATOR_DIRS) + " |     pooled"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        row = f"{name:<24} | "
        for g in GENERATOR_DIRS:
            e = res["per_generator"].get(g, {}).get(qk, {}).get("enrichment", float("nan"))
            row += f"{e:>10.2f} | "
        row += f"{res['pooled'][qk]['enrichment']:>10.2f}"
        print(row)

    # 8. Критерий прохождения шага 2
    print("\n===== Критерий шага 2 =====")
    primary = results.get("fold_logit_hybrid", {})
    passed = 0
    best_naive = {}
    for g in GENERATOR_DIRS:
        e_primary = primary.get("per_generator", {}).get(g, {}).get(qk, {}).get("enrichment", float("nan"))
        naive = {
            n: results[n]["per_generator"].get(g, {}).get(qk, {}).get("enrichment", float("nan"))
            for n in ["clash_pairs", "length", "random"]
        }
        best_naive[g] = max(naive.values())
        ok = e_primary >= 2.0 and e_primary > best_naive[g]
        passed += int(ok)
        print(f"  {g}: fold_logit_hybrid {e_primary:.2f}× vs лучший наивный "
              f"{best_naive[g]:.2f}× → {'PASS' if ok else 'FAIL'}")
    print(f"Итог: {passed}/5 генераторов (нужно >= 3 с обогащением >= 2× и выше наивного бейзлайна)")
    print(f"\nСохранено: {os.path.abspath(OUT_CSV)}, {os.path.abspath(OUT_JSON)}")


if __name__ == "__main__":
    main()
