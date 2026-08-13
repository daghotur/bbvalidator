"""
analysis_baselines.py
---------------------
Батарея дешёвых бейзлайнов против обученной модели: бьём ли мы вообще что-то,
кроме тривиальных геометрических дескрипторов?

Проверяются три класса, различающиеся ценой:

  БЕСПЛАТНЫЕ (из одних координат, как и наша модель):
    * rg_norm      — радиус инерции, нормированный на N^(1/3): компактность;
    * contact_order — относительный контактный порядок (Plaxco): классический
                      предиктор скорости фолдинга, топологическая сложность;
    * packing      — средняя плотность контактов CA в радиусе 8 A;
    * length       — длина цепи (проверка на длиновой шорткат).

  ДЕШЁВЫЙ, НО НЕ БЕСПЛАТНЫЙ (нужен прогон ProteinMPNN, ~мс на остов):
    * mpnn_global  — средний global_score: NLL по всем остаткам;
    * mpnn_design  — средний score: NLL только по дизайнируемым позициям
                     (мотив зафиксирован), из поля header.

  НАША МОДЕЛЬ — один проход, координаты на входе.

Бейзлайнам дана фора: направление ранжирования выбирается по лучшему из двух
на каждом генераторе (оптимистичная оценка), наша модель идёт в своём
естественном направлении. Если мы всё равно выигрываем — вывод устойчив.

Метрика и потолок — как в analysis_relabel.py: внутримотивно, метка min по 8
последовательностям, потолок оракула на общей половине B.

Запуск:  python analysis_baselines.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from analysis_label_choice import AGGREGATORS, MIN_SAMPLES_PER_MOTIF, N_SPLITS, TOP_FRAC
from analysis_logits_ranking import EVAL_SOURCES, GENERATOR_DIRS, parse_pdb_files
from analysis_oracle_ceiling import (
    DESIGNABLE_MAX_SCRMSD,
    SATURATED_HIGH,
    SATURATED_LOW,
    load_per_sequence_rmsd,
)
from analysis_perresidue import score
from analysis_scrmsd import SCRMSD_AGG
from inference import build_model
from train_soft import HOLDOUT_GENERATORS

CONTACT_CUTOFF = 8.0
MIN_SEQ_SEP = 3
RNG_SEED = 42
OUT_JSON = "analysis_baselines.json"


def geometric_descriptors(coords: np.ndarray) -> dict:
    """Дескрипторы из одних координат CA. coords: [L, 3, 3] (N, CA, C)."""
    ca = coords[:, 1, :].astype(np.float64)
    L = len(ca)
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)

    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))

    idx = np.arange(L)
    sep = np.abs(idx[:, None] - idx[None, :])
    contact = (d < CONTACT_CUTOFF) & (sep >= MIN_SEQ_SEP)
    n_contacts = int(np.triu(contact).sum())
    # относительный контактный порядок (Plaxco): средняя |i-j| контактов / L
    co = float(np.triu(contact * sep).sum() / (L * n_contacts)) if n_contacts else 0.0

    return {
        "rg_norm": rg / (L ** (1.0 / 3.0)),
        "contact_order": co,
        "packing": float(contact.sum(1).mean()),
        "length": float(L),
    }


def parse_mpnn(root: str) -> dict:
    """(motif, sample) -> средние ProteinMPNN NLL: по всем остаткам и по дизайну."""
    out = {}
    for path in glob.glob(os.path.join(root, "**", "esm_eval_results.csv"), recursive=True):
        if "__MACOSX" in path:
            continue
        parts = path.split(os.sep)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "mpnn_score" not in df.columns or "header" not in df.columns:
            continue
        # header: "T=0.1, sample=1, score=1.0111, global_score=1.3253, ..."
        design = df["header"].str.extract(r"score=([\d.]+)")[0]
        design = pd.to_numeric(design, errors="coerce")
        glob_ = pd.to_numeric(df["mpnn_score"], errors="coerce")
        if glob_.notna().sum() == 0:
            continue
        out[(parts[-4], parts[-3])] = {
            "mpnn_global": float(glob_.mean()),
            "mpnn_design": float(design.mean()) if design.notna().any() else np.nan,
        }
    return out


def precision_at_top(rank_by: np.ndarray, design: np.ndarray) -> float:
    k = max(1, round(TOP_FRAC * len(design)))
    return float(design[np.argsort(rank_by)[:k]].mean())


def evaluate(values: np.ndarray, samples: list[np.ndarray], agg, rng,
             best_direction: bool) -> dict | None:
    """Ранжирующее качество произвольного скора на одном мотиве."""
    if np.isnan(values).any() or len(np.unique(values)) < 3:
        return None
    full = np.array([agg(s) for s in samples])
    design = full < DESIGNABLE_MAX_SCRMSD
    base = design.mean()
    if not (SATURATED_LOW <= base <= SATURATED_HIGH):
        return None

    sp = spearmanr(values, full)[0]
    # фора бейзлайнам: берём то направление, которое лучше на этом мотиве
    signs = (1.0, -1.0) if best_direction else (1.0,)

    m_lifts, o_lifts = [], []
    for _ in range(N_SPLITS):
        a, b = [], []
        for s in samples:
            i = rng.permutation(len(s))
            h = len(s) // 2
            a.append(agg(s[i[:h]]))
            b.append(agg(s[i[h:]]))
        a, b = np.array(a), np.array(b)
        d_b = b < DESIGNABLE_MAX_SCRMSD
        if d_b.all() or not d_b.any():
            continue
        m_lifts.append(max(precision_at_top(sg * values, d_b) for sg in signs) / d_b.mean())
        o_lifts.append(precision_at_top(a, d_b) / d_b.mean())
    if not m_lifts:
        return None
    return {
        "spearman": float(sp),
        "lift": float(np.mean(m_lifts)),
        "ceiling": float(np.mean(o_lifts)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agg = AGGREGATORS[SCRMSD_AGG]
    model = build_model("checkpoints/joint_model.pth", device, per_residue=True)

    geo_names = ["rg_norm", "contact_order", "packing", "length"]
    mpnn_names = ["mpnn_global", "mpnn_design"]
    scorers = geo_names + mpnn_names + ["наша модель"]

    results = {}
    for gen in GENERATOR_DIRS:
        records, _ = parse_pdb_files(GENERATOR_DIRS[gen])
        per_seq = load_per_sequence_rmsd(EVAL_SOURCES[gen])
        mpnn = parse_mpnn(EVAL_SOURCES[gen])
        pred = score(model, records, device, "fold")
        model_lookup = dict(zip(zip(pred["motif"], pred["sample"]), pred["pred_scrmsd"]))

        by_motif: dict[str, list] = {}
        for r in records:
            key = (r["motif"], r["sample"].removesuffix(".pdb"))
            if key not in per_seq or key not in mpnn or key not in model_lookup:
                continue
            row = geometric_descriptors(r["coords"])
            row.update(mpnn[key])
            row["наша модель"] = model_lookup[key]
            by_motif.setdefault(r["motif"], []).append((row, per_seq[key]))

        res = {}
        for name in scorers:
            rng = np.random.default_rng(RNG_SEED)
            rows = []
            for motif, items in by_motif.items():
                if len(items) < MIN_SAMPLES_PER_MOTIF:
                    continue
                v = np.array([x[0][name] for x in items], dtype=float)
                s = [x[1] for x in items]
                r = evaluate(v, s, agg, rng, best_direction=(name != "наша модель"))
                if r:
                    rows.append(r)
            ms = pd.DataFrame(rows)
            res[name] = {
                "n_kept": int(len(ms)),
                "spearman": float(ms["spearman"].median()) if len(ms) else None,
                "abs_spearman": float(ms["spearman"].abs().median()) if len(ms) else None,
                "lift": float(ms["lift"].median()) if len(ms) else None,
                "frac_ceiling": float((ms["lift"] / ms["ceiling"]).median()) if len(ms) else None,
            }
        results[gen] = res

    hold = set(HOLDOUT_GENERATORS)
    print(f"\n{'генератор':16} {'скорер':15} {'|Spearman|':>10} {'lift':>7} {'доля потолка':>13}")
    print("-" * 68)
    for gen, res in results.items():
        print(f"{gen}{'  ХОЛДАУТ' if gen in hold else ''}")
        for name in scorers:
            v = res[name]
            if v["lift"] is None:
                print(f"{'':16} {name:15} {'—':>10} {'—':>7} {'—':>13}")
                continue
            mark = " ←" if name == "наша модель" else ""
            print(f"{'':16} {name:15} {v['abs_spearman']:>10.3f} "
                  f"{v['lift']:>6.2f}x {v['frac_ceiling']:>12.0%}{mark}")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}")


if __name__ == "__main__":
    main()
