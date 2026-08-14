"""
filter_designability.py
-----------------------
Фильтрация структур по дизайнуемости: для каждого backbone выдаёт вердикт
PASS/FAIL и показывает, какие конкретно метрики завалены.

Две модели с разделёнными ролями (не смешивать в один смысловой скор):

  * Ранжирование и вердикт — pure soft (`soft_model.pth`, дообучение на
    log1p(scRMSD)): детерминированный предсказанный scRMSD в Å, меньше =
    дизайнируемее. Только эта модель определяет PASS/FAIL и порядок
    при фильтрации потока.
  * Диагностика похожести на натив — soft_mt (`soft_model_mt.pth`,
    мультизадачное дообучение с native-якорем): предсказанный scRMSD из
    софта с нативной калибровкой, RMSD aux-головы, стерика, failure mode.
    Эти колонки помечены отдельно и НИКОГДА не участвуют в вердикте.

Расхождение двух моделей — не ошибка, а сигнал: «хорошо ранжируется, но не
похоже на натив» — именно тот случай, который фильтр должен подсвечивать.

Обязательный гейт — --max-scrmsd (по ранжирующей модели). Опциональные:
--max-clashes (сырые конфликты геометрии, модельно-независимые) и
--max-uncertainty (при --mc-runs > 1).

Единицы колонки uncertainty: дисперсия предсказанного log1p(scRMSD) по
MC-проходам. В evaluation/eval_generated.py колонка с тем же именем — это
дисперсия ВЕРОЯТНОСТИ бинарной модели (≤ 0.25). Числа из двух артефактов
несравнимы между собой.

Примеры:
    python filter_designability.py -i data/ood/evodiff/scaffolds/01_1LDB
    python filter_designability.py -i data/ood/rfdiffusion/scaffolds --max-clashes 2 -o rfdiff_filter.csv
    python filter_designability.py -i data/3MYC.pdb --mc-runs 8
"""

import argparse
import os

import numpy as np
import torch

from common.structures import collect_pdbs, iter_padded_batches
from inference import _autocast_ctx, build_model, center_coords, parse_pdb_to_backbone
from model.heads_loss import mc_fold_logits

# Классы failure_mode по FAILURE_MODE_MAP из dataset/build_negative_dataset.py
FAILURE_MODE_NAMES = [
    "ok",                # 0: positive_real
    "easy",              # 1: глобальный шум / разрыв цепи
    "hard",              # 2: распакованное ядро / ложная компактность
    "near_native",       # 3
    "borderline",        # 4: дефект хинджа / поворот фрагмента
    "unknown",           # 5: зарезервирован под OOD-негативы
]
DIAG_NOT_NATIVE_LIKE_A = 4.0  # маркер расхождения: PASS по рангу, но диагностика далеко от натива


@torch.no_grad()
def score_batch(
    model, diag_model, coords: torch.Tensor, mask: torch.Tensor,
    device: torch.device, mc_runs: int,
) -> dict:
    """Ранжирующая модель + диагностическая модель + сырая геометрия."""
    # Сырые геометрические диагностики — в float32, без автокаста
    geom = model.frontend.geometry(coords, mask)
    m = mask.float()
    clash_pairs = (geom["clash_count"] * m).sum(dim=-1) / 2.0  # симметричная матрица
    hbonds = (geom["hbond_count"] * m).sum(dim=-1)

    with _autocast_ctx(device):
        # 1) Ранжирование и вердикт — только pure soft
        preds = model(coords, mask)
        pred_scrmsd = torch.clamp(torch.expm1(preds["fold_logit"]).float(), min=0.0)

        # 2) Диагностика похожести на натив — только soft_mt
        dp = diag_model(coords, mask)
        diag_similarity = torch.clamp(torch.expm1(dp["fold_logit"]).float(), min=0.0)
        rmsd_aux = torch.clamp(torch.expm1(dp["rmsd"]).float(), min=0.0)
        steric = torch.sigmoid(dp["steric"]).float()
        fm_logits = dp["failure_mode"].float()

        # 3) Неопределённость ранжирующей модели (опционально).
        # Физика внутри считается один раз на все проходы (см. mc_fold_logits).
        uncertainty = torch.zeros(coords.shape[0], device=device)
        if mc_runs > 1:
            uncertainty = mc_fold_logits(model, coords, mask, mc_runs).float().var(dim=0)

    fm_prob = torch.softmax(fm_logits, dim=-1)
    fm_p, fm_idx = fm_prob.max(dim=-1)

    return {
        "pred_scrmsd": pred_scrmsd.cpu().numpy(),
        "uncertainty": uncertainty.cpu().numpy(),
        "diag_similarity": diag_similarity.cpu().numpy(),
        "p_steric": steric.cpu().numpy(),
        "rmsd_aux": rmsd_aux.cpu().numpy(),
        "fm_idx": fm_idx.cpu().numpy(),
        "fm_p": fm_p.cpu().numpy(),
        "clash_pairs": clash_pairs.cpu().numpy(),
        "hbonds": hbonds.cpu().numpy(),
    }


def gates_failed(r: dict, args) -> list[str]:
    reasons = []
    if r["pred_scrmsd"] > args.max_scrmsd:
        reasons.append(f"pred scRMSD {r['pred_scrmsd']:.2f} Å > {args.max_scrmsd:.2f} Å")
    if args.max_clashes is not None and r["clash_pairs"] > args.max_clashes:
        reasons.append(f"конфликты {int(r['clash_pairs'])} > {args.max_clashes}")
    if args.max_uncertainty is not None and r["uncertainty"] > args.max_uncertainty:
        reasons.append(f"uncertainty {r['uncertainty']:.4f} > {args.max_uncertainty:.4f}")
    return reasons


def main():
    parser = argparse.ArgumentParser(
        description="Фильтрация структур по дизайнуемости с подробным вердиктом"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="PDB-файл или директория со структурами")
    parser.add_argument("--pattern", default="**/*.pdb",
                        help="glob-паттерн внутри директории")
    parser.add_argument("-c", "--ckpt", default="checkpoints/soft_model.pth",
                        help="ранжирующая модель (вердикт)")
    parser.add_argument("--diag-ckpt", default="checkpoints/soft_model_mt.pth",
                        help="диагностическая модель (похожесть на натив, не влияет на вердикт)")
    parser.add_argument("--pca", default="dataset/pca_components.pth")
    parser.add_argument("-m", "--mc-runs", type=int, default=1,
                        help="MC-проходы для неопределённости (1 = детерминированный скор)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-scrmsd", type=float, default=2.0,
                        help="обязательный гейт по предсказанному scRMSD ранжирующей модели, Å")
    parser.add_argument("--max-clashes", type=int, default=None,
                        help="гейт по сырому числу стерических конфликтов (Cβ < 3.5 Å, |i-j| >= 3)")
    parser.add_argument("--max-uncertainty", type=float, default=None,
                        help="гейт по MC-дисперсии предсказанного log1p(scRMSD) "
                             "(только при --mc-runs > 1; это НЕ дисперсия вероятности, "
                             "как в evaluation/eval_generated.py)")
    parser.add_argument("-o", "--output", default="results/filter_results.csv")
    parser.add_argument("--quiet", action="store_true",
                        help="не печатать построчный вердикт, только сводку")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Устройство: {device} | mc_runs={args.mc_runs} | batch={args.batch_size}")

    files = collect_pdbs(args.input, args.pattern)
    if not files:
        print(f"Не найдено PDB по пути {args.input} (паттерн {args.pattern})")
        return
    print(f"Структур к скорингу: {len(files)}")

    print("Ранжирующая модель (вердикт):")
    model = build_model(args.ckpt, device, pca_path=args.pca)
    print("Диагностическая модель (похожесть на натив):")
    diag_model = build_model(args.diag_ckpt, device, pca_path=args.pca)

    # Парсинг
    records, skipped = [], 0
    for f in files:
        try:
            coords = center_coords(parse_pdb_to_backbone(f)).astype(np.float32)
        except Exception as e:
            if not args.quiet:
                print(f"ПРОПУСК {os.path.basename(f)}: {e}")
            skipped += 1
            continue
        records.append({"path": f, "coords": coords})
    print(f"Распарсено: {len(records)} (пропущено: {skipped})")

    # Батчевый скоринг с сортировкой по длине (меньше паддинг)
    rows: list[dict] = []
    for idxs, coords, mask in iter_padded_batches(
        records, args.batch_size, device
    ):
        res = score_batch(model, diag_model, coords, mask, device, args.mc_runs)
        for b, i in enumerate(idxs):
            r = {k: v[b] for k, v in res.items()}
            r["path"] = records[i]["path"]
            r["length"] = len(records[i]["coords"])
            r["reasons"] = gates_failed(r, args)
            rows.append(r)

    rows.sort(key=lambda r: r["pred_scrmsd"])

    # Построчный вердикт
    if not args.quiet:
        print()
        print(f"{'Файл':<28} | {'L':>4} | {'scRMSD':>6} | {'u':>7} | {'натив':>5} | "
              f"{'RMSD_aux':>8} | {'p_стер':>6} | {'failure_mode':<13} | {'конфл':>5} | вердикт")
        print("-" * 130)
        for r in rows:
            name = os.path.basename(r["path"])[:26]
            fm = f"{FAILURE_MODE_NAMES[r['fm_idx']]}({r['fm_p']:.2f})"
            if r["reasons"]:
                verdict = "FAIL: " + "; ".join(r["reasons"])
            else:
                verdict = "PASS"
                if r["diag_similarity"] > DIAG_NOT_NATIVE_LIKE_A:
                    verdict += " (низкая похожесть на натив)"
            print(f"{name:<28} | {r['length']:>4} | {r['pred_scrmsd']:>6.2f} | "
                  f"{r['uncertainty']:>7.4f} | {r['diag_similarity']:>5.2f} | {r['rmsd_aux']:>8.2f} | "
                  f"{r['p_steric']:>6.3f} | {fm:<13} | {int(r['clash_pairs']):>5} | {verdict}")

    # CSV
    with open(args.output, "w", encoding="utf-8") as fp:
        fp.write("name,length,pred_scrmsd,uncertainty,diag_similarity,p_steric,rmsd_aux,"
                 "failure_mode,failure_mode_p,clash_pairs,hbond_count,verdict,reasons\n")
        for r in rows:
            verdict = "FAIL" if r["reasons"] else "PASS"
            reasons = "; ".join(r["reasons"])
            fp.write(
                f"{os.path.basename(r['path'])},{r['length']},{r['pred_scrmsd']:.4f},"
                f"{r['uncertainty']:.6f},{r['diag_similarity']:.4f},{r['p_steric']:.4f},"
                f"{r['rmsd_aux']:.3f},{FAILURE_MODE_NAMES[r['fm_idx']]},{r['fm_p']:.4f},"
                f"{int(r['clash_pairs'])},{int(r['hbonds'])},{verdict},{reasons}\n"
            )

    # Сводка
    n_pass = sum(1 for r in rows if not r["reasons"])
    n_disagree = sum(
        1 for r in rows
        if not r["reasons"] and r["diag_similarity"] > DIAG_NOT_NATIVE_LIKE_A
    )
    print()
    print("=" * 60)
    print(f"Гейты: max_scrmsd={args.max_scrmsd}"
          + (f" | max_clashes={args.max_clashes}" if args.max_clashes is not None else "")
          + (f" | max_uncertainty={args.max_uncertainty}" if args.max_uncertainty is not None else ""))
    print(f"PASS: {n_pass}/{len(rows)}"
          + (f" (пропущено при парсинге: {skipped})" if skipped else "")
          + (f" | из них с расхождением «ранжируется, но не похоже на натив»: {n_disagree}" if n_disagree else ""))
    if rows:
        p = np.array([r["pred_scrmsd"] for r in rows])
        print(f"pred scRMSD: mean {p.mean():.2f} | median {np.median(p):.2f} "
              f"| min {p.min():.2f} | max {p.max():.2f} Å")
    print(f"CSV: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
