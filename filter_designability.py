"""
filter_designability.py
-----------------------
Фильтрация структур по дизайнуемости: для каждого backbone выдаёт вердикт
PASS/FAIL и показывает, какие конкретно метрики завалены.

Обязательный гейт: MC-Dropout P(fold) >= --min-pfold. Опциональные гейты:
вероятность стерики (--max-steric), сырое число стерических конфликтов
(--max-clashes), предсказанный RMSD (--max-rmsd), эпистемическая
неопределённость (--max-uncertainty). Неподключённые гейты (None) на вердикт
не влияют, но их значения всегда видны в verbose-выводе и CSV.

Примеры:
    python filter_designability.py -i data/ --min-pfold 0.5
    python filter_designability.py -i data/ood/evodiff --max-clashes 2 -o evodiff_filter.csv
    python filter_designability.py -i data/3MYC.pdb -m 32
"""

import argparse
import glob
import os

import numpy as np
import torch

from inference import _autocast_ctx, build_model, center_coords, parse_pdb_to_backbone
from model.heads_loss import predict_with_uncertainty

# Классы failure_mode по FAILURE_MODE_MAP из dataset/build_negative_dataset.py
FAILURE_MODE_NAMES = [
    "ok",                # 0: positive_real
    "easy",              # 1: глобальный шум / разрыв цепи
    "hard",              # 2: распакованное ядро / ложная компактность
    "near_native",       # 3
    "borderline",        # 4: дефект хинджа / поворот фрагмента
    "unknown",           # 5: зарезервирован под OOD-негативы
]


def collect_pdbs(input_path: str, pattern: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    files = sorted(glob.glob(os.path.join(input_path, pattern), recursive=True))
    return [
        f
        for f in files
        if "__MACOSX" not in f and not os.path.basename(f).startswith("._")
    ]


@torch.no_grad()
def score_batch(
    model, coords: torch.Tensor, mask: torch.Tensor, device: torch.device, mc_runs: int
) -> dict:
    """Все метрики одного батча: головы модели + сырая геометрия."""
    # Сырые геометрические диагностики — в float32, без автокаста
    geom = model.frontend.geometry(coords, mask)
    m = mask.float()
    clash_pairs = (geom["clash_count"] * m).sum(dim=-1) / 2.0  # симметричная матрица
    hbonds = (geom["hbond_count"] * m).sum(dim=-1)

    with _autocast_ctx(device):
        mc = predict_with_uncertainty(model, coords, mask, mc_runs=mc_runs)
        preds = model(coords, mask)

    steric = torch.sigmoid(preds["steric"]).float()
    fm_logits = preds["failure_mode"].float()
    fm_prob = torch.softmax(fm_logits, dim=-1)
    fm_p, fm_idx = fm_prob.max(dim=-1)

    return {
        "pfold": mc["p_foldable"].float().cpu().numpy(),
        "uncertainty": mc["uncertainty"].float().cpu().numpy(),
        "p_steric": steric.cpu().numpy(),
        "rmsd": torch.clamp(torch.expm1(preds["rmsd"]).float(), min=0.0).cpu().numpy(),
        "fm_idx": fm_idx.cpu().numpy(),
        "fm_p": fm_p.cpu().numpy(),
        "clash_pairs": clash_pairs.cpu().numpy(),
        "hbonds": hbonds.cpu().numpy(),
    }


def gates_failed(r: dict, args) -> list[str]:
    reasons = []
    if r["pfold"] < args.min_pfold:
        reasons.append(f"pfold {r['pfold']:.3f} < {args.min_pfold:.2f}")
    if args.max_steric is not None and r["p_steric"] > args.max_steric:
        reasons.append(f"стерика p {r['p_steric']:.3f} > {args.max_steric:.2f}")
    if args.max_clashes is not None and r["clash_pairs"] > args.max_clashes:
        reasons.append(
            f"конфликты {int(r['clash_pairs'])} > {args.max_clashes}"
        )
    if args.max_rmsd is not None and r["rmsd"] > args.max_rmsd:
        reasons.append(f"rmsd {r['rmsd']:.2f} Å > {args.max_rmsd:.2f} Å")
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
    parser.add_argument("-c", "--ckpt", default="checkpoints/best_model.pth")
    parser.add_argument("--pca", default="dataset/pca_components.pth")
    parser.add_argument("-m", "--mc-runs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-pfold", type=float, default=0.5,
                        help="обязательный гейт по P(fold)")
    parser.add_argument("--max-steric", type=float, default=None,
                        help="гейт по вероятности стерики (sigmoid steric-головы)")
    parser.add_argument("--max-clashes", type=int, default=None,
                        help="гейт по сырому числу стерических конфликтов (Cβ < 3.5 Å, |i-j| >= 3)")
    parser.add_argument("--max-rmsd", type=float, default=None,
                        help="гейт по предсказанному RMSD, Å")
    parser.add_argument("--max-uncertainty", type=float, default=None,
                        help="гейт по эпистемической неопределённости (дисперсия MC)")
    parser.add_argument("-o", "--output", default="filter_results.csv")
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

    model = build_model(args.ckpt, device, pca_path=args.pca)

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
    order = sorted(range(len(records)), key=lambda i: len(records[i]["coords"]))
    rows: list[dict] = []
    for start in range(0, len(order), args.batch_size):
        idxs = order[start : start + args.batch_size]
        Lmax = max(len(records[i]["coords"]) for i in idxs)
        B = len(idxs)
        coords = np.zeros((B, Lmax, 3, 3), dtype=np.float32)
        mask = np.zeros((B, Lmax), dtype=np.bool_)
        for b, i in enumerate(idxs):
            L = len(records[i]["coords"])
            coords[b, :L] = records[i]["coords"]
            mask[b, :L] = True
        res = score_batch(
            model,
            torch.from_numpy(coords).to(device),
            torch.from_numpy(mask).to(device),
            device,
            args.mc_runs,
        )
        for b, i in enumerate(idxs):
            r = {k: v[b] if not isinstance(v[b], np.ndarray) else v[b] for k, v in res.items()}
            r["path"] = records[i]["path"]
            r["length"] = len(records[i]["coords"])
            r["reasons"] = gates_failed(r, args)
            rows.append(r)

    rows.sort(key=lambda r: r["pfold"])

    # Построчный вердикт
    if not args.quiet:
        print()
        print(f"{'Файл':<28} | {'L':>4} | {'P(fold)':>7} | {'u':>7} | {'p_стер':>6} | "
              f"{'RMSD':>5} | {'конфл':>5} | {'H-св':>4} | {'failure_mode':<13} | вердикт")
        print("-" * 118)
        for r in rows:
            name = os.path.basename(r["path"])[:26]
            fm = f"{FAILURE_MODE_NAMES[r['fm_idx']]}({r['fm_p']:.2f})"
            if r["reasons"]:
                verdict = "FAIL: " + "; ".join(r["reasons"])
            else:
                verdict = "PASS"
            print(f"{name:<28} | {r['length']:>4} | {r['pfold']:>7.3f} | "
                  f"{r['uncertainty']:>7.4f} | {r['p_steric']:>6.3f} | {r['rmsd']:>5.2f} | "
                  f"{int(r['clash_pairs']):>5} | {int(r['hbonds']):>4} | {fm:<13} | {verdict}")

    # CSV
    with open(args.output, "w", encoding="utf-8") as fp:
        fp.write("name,length,pfold,uncertainty,p_steric,rmsd_pred,failure_mode,"
                 "failure_mode_p,clash_pairs,hbond_count,verdict,reasons\n")
        for r in rows:
            verdict = "FAIL" if r["reasons"] else "PASS"
            reasons = "; ".join(r["reasons"])
            fp.write(
                f"{os.path.basename(r['path'])},{r['length']},{r['pfold']:.4f},"
                f"{r['uncertainty']:.6f},{r['p_steric']:.4f},{r['rmsd']:.3f},"
                f"{FAILURE_MODE_NAMES[r['fm_idx']]},{r['fm_p']:.4f},"
                f"{int(r['clash_pairs'])},{int(r['hbonds'])},{verdict},{reasons}\n"
            )

    # Сводка
    n_pass = sum(1 for r in rows if not r["reasons"])
    print()
    print("=" * 60)
    print(f"Гейты: min_pfold={args.min_pfold}"
          + (f" | max_steric={args.max_steric}" if args.max_steric is not None else "")
          + (f" | max_clashes={args.max_clashes}" if args.max_clashes is not None else "")
          + (f" | max_rmsd={args.max_rmsd}" if args.max_rmsd is not None else "")
          + (f" | max_uncertainty={args.max_uncertainty}" if args.max_uncertainty is not None else ""))
    print(f"PASS: {n_pass}/{len(rows)}"
          + (f" (пропущено при парсинге: {skipped})" if skipped else ""))
    if rows:
        p = np.array([r["pfold"] for r in rows])
        print(f"P(fold): mean {p.mean():.3f} | median {np.median(p):.3f} "
              f"| min {p.min():.3f} | max {p.max():.3f}")
    print(f"CSV: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
