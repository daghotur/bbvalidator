"""
analysis_length_stress.py
-------------------------
Стресс-тест архитектуры по длине цепи на данных Scaffold-Lab.

Зачем. Всё обучение и вся валидация проекта — остова 75-200 остатков из MotifBench.
Между тем в архитектуре есть решения, привязанные к масштабу и никогда не проверявшиеся
за его пределами:

  * в трансформере нет позиционного кодирования — порядок остатков доходит только
    через 2 графовых слоя;
  * граф строится по k=16 ближайшим соседям независимо от длины: на 1000 остатков
    это вдвое меньшая доля цепи, чем на 100;
  * relative_burial нормируется на N^(1/3) и обрезается на 2.0;
  * attention-пулинг берёт softmax по всей длине — на длинной цепи он может выродиться
    либо в равномерный, либо в несколько остатков.

Scaffold-Lab (Zenodo 20080699) даёт безусловно сгенерированные остова длиной 50-1000
от пяти генераторов, четыре из которых модель не видела. Меток self-consistency там
нет, поэтому проверяются свойства, не требующие разметки:

  1. НЕ ВЫРОЖДАЕТСЯ ЛИ СКОР — если внутри группы (генератор, длина) разброс предсказаний
     схлопывается, ранжировать нечем в принципе, независимо от точности;
  2. ПРАВИЛЬНО ЛИ НАПРАВЛЕНИЕ — в литературе дизайнируемость падает с длиной, значит
     предсказанный scRMSD обязан расти; инверсия означала бы поломку;
  3. НЕ НАСЫЩАЮТСЯ ЛИ ПРИЗНАКИ — доля остатков, упирающихся в clamp;
  4. НЕ ВЫРОЖДАЕТСЯ ЛИ ПУЛИНГ — энтропия attention-весов относительно равномерной.

Запуск:  python analysis_length_stress.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from inference import _autocast_ctx, build_model, center_coords, parse_pdb_to_backbone

ROOT = "data/ood/scaffold_lab/original_scaffolds"
CKPT = "checkpoints/joint_model.pth"
OUT_JSON = "analysis_length_stress.json"
OUT_CSV = "length_stress_per_sample.csv"
ATTN_BUDGET = 8_000_000   # B * N^2, чтобы не упереться в память на длинных цепях


def collect() -> list[dict]:
    items = []
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "length_*", "*.pdb"))):
        if "__MACOSX" in path:
            continue
        parts = path.split(os.sep)
        items.append({
            "generator": parts[-3],
            "length_bin": int(parts[-2].removeprefix("length_")),
            "name": os.path.basename(path),
            "path": path,
        })
    return items


@torch.no_grad()
def score_group(model, group: list[dict], device) -> list[dict]:
    """Скоринг одной группы одинаковой номинальной длины."""
    coords_list, kept = [], []
    for it in group:
        try:
            coords_list.append(center_coords(parse_pdb_to_backbone(it["path"])))
            kept.append(it)
        except Exception:
            continue
    if not kept:
        return []

    N = max(len(c) for c in coords_list)
    B = max(1, min(32, int(ATTN_BUDGET / max(N, 1) ** 2)))

    out = []
    for s in range(0, len(kept), B):
        chunk = coords_list[s : s + B]
        Lmax = max(len(c) for c in chunk)
        b = len(chunk)
        arr = np.zeros((b, Lmax, 3, 3), dtype=np.float32)
        msk = np.zeros((b, Lmax), dtype=np.bool_)
        for i, c in enumerate(chunk):
            arr[i, : len(c)] = c
            msk[i, : len(c)] = True
        ct = torch.from_numpy(arr).to(device)
        mt = torch.from_numpy(msk).to(device)

        feats = model.compute_features(ct, mt)
        with _autocast_ctx(device):
            preds = model(ct, mt)

        pred = torch.clamp(torch.expm1(preds["fold_logit"].float()), min=0.0)
        lddt = torch.sigmoid(preds["lddt_logit"].float())
        attn = preds["attn_weights"].float()          # [B, N, H]
        mf = mt.float()

        # энтропия attention относительно равномерной по валидным позициям
        w = attn.mean(-1)                              # усредняем по головам
        w = w * mf
        w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
        ent = -(w * torch.log(w.clamp(min=1e-12))).sum(1)
        n_valid = mf.sum(1)
        ent_ratio = ent / torch.log(n_valid.clamp(min=2.0))

        # насыщение признаков
        des = model.frontend.designability(ct[:, :, 1, :], mt)
        burial_sat = ((des["relative_burial"] >= 2.0).float() * mf).sum(1) / n_valid
        p10 = (des["packing_r10"] >= 50.0).float().mul(mf).sum(1) / n_valid

        mean_lddt = (lddt * mf).sum(1) / n_valid

        for i, it in enumerate(kept[s : s + B]):
            out.append({
                **{k: it[k] for k in ("generator", "length_bin", "name")},
                "L": int(n_valid[i].item()),
                "pred_scrmsd": float(pred[i]),
                "mean_lddt": float(mean_lddt[i]),
                "attn_entropy_ratio": float(ent_ratio[i]),
                "burial_saturated": float(burial_sat[i]),
                "packing_saturated": float(p10[i]),
            })
    return out


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(f"нет {ROOT} — скачайте Scaffold-Lab (Zenodo 20080699)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(CKPT, device, per_residue=True)

    items = collect()
    print(f"Найдено {len(items)} остовов, генераторов {len({i['generator'] for i in items})}")

    by_group: dict[tuple, list] = {}
    for it in items:
        by_group.setdefault((it["generator"], it["length_bin"]), []).append(it)

    rows = []
    for (gen, L), grp in sorted(by_group.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows += score_group(model, grp, device)
        print(f"  {gen:12} L={L:<5} готово ({len(grp)})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n=== 1. РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ ПО ДЛИНЕ ===")
    print(f"{'длина':>6} {'n':>5} {'медиана':>9} {'IQR':>16} {'std':>7} "
          f"{'дизайн.<2Å':>11} {'уник. знач.':>12}")
    print("-" * 74)
    length_tab = {}
    for L, g in df.groupby("length_bin"):
        q1, q3 = g["pred_scrmsd"].quantile([0.25, 0.75])
        length_tab[int(L)] = {
            "n": int(len(g)),
            "median": float(g["pred_scrmsd"].median()),
            "iqr": [float(q1), float(q3)],
            "std": float(g["pred_scrmsd"].std()),
            "frac_designable": float((g["pred_scrmsd"] < 2.0).mean()),
            "n_unique": int(g["pred_scrmsd"].round(3).nunique()),
        }
        print(f"{L:>6} {len(g):>5} {g['pred_scrmsd'].median():>9.2f} "
              f"[{q1:>6.2f},{q3:>6.2f}] {g['pred_scrmsd'].std():>7.2f} "
              f"{(g['pred_scrmsd'] < 2.0).mean():>10.1%} {length_tab[int(L)]['n_unique']:>12}")

    print("\n=== 2. ПО ГЕНЕРАТОРАМ (медианный предсказанный scRMSD) ===")
    piv = df.pivot_table(index="generator", columns="length_bin",
                         values="pred_scrmsd", aggfunc="median")
    print(piv.round(2).to_string())

    print("\n=== 3. НАСЫЩЕНИЕ ПРИЗНАКОВ И ПУЛИНГА ===")
    print(f"{'длина':>6} {'burial ≥2.0':>12} {'packing ≥50':>12} {'энтропия attn':>14}")
    print("-" * 48)
    sat_tab = {}
    for L, g in df.groupby("length_bin"):
        sat_tab[int(L)] = {
            "burial_saturated": float(g["burial_saturated"].mean()),
            "packing_saturated": float(g["packing_saturated"].mean()),
            "attn_entropy_ratio": float(g["attn_entropy_ratio"].mean()),
        }
        print(f"{L:>6} {g['burial_saturated'].mean():>11.1%} "
              f"{g['packing_saturated'].mean():>11.1%} "
              f"{g['attn_entropy_ratio'].mean():>13.3f}")
    print("\n(энтропия: 1.0 = равномерное внимание по всей цепи, 0 = всё в один остаток)")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump({"by_length": length_tab, "saturation": sat_tab,
                   "median_by_generator": piv.round(4).to_dict()},
                  fp, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {OUT_JSON}, {OUT_CSV}")


if __name__ == "__main__":
    main()
