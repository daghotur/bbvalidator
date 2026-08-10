"""
eval_generated.py
-----------------
OOD-оценка архитектур на выходах внешних генераторов (MotifBench-скаффолды):
структуры, которых не было ни в обучении, ни в синтетических декоях.

Для каждого чекпоинта (гибрид / MLP / GPS) считается MC-Dropout
P(foldable) и эпистемическая неопределённость по всем структурам каждого
генератора, плюс референсные группы native_test / decoy_test из нашего
test-сплита (чтобы видеть, где генераторы на шкале «натив ↔ декоев»).

Запуск:
    python eval_generated.py                       # полный прогон
    python eval_generated.py --max-per-generator 30 --mc-runs 4   # смоук
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from dataset.dataloader import make_loader
from eval_model import build_eval_model
from inference import _autocast_ctx, center_coords, parse_pdb_to_backbone
from model.heads_loss import predict_with_uncertainty

GENERATOR_DIRS = {
    "RFdiffusion": "data/ood/rfdiffusion",
    "RFdiffusion-AA": "data/ood/rfdiffusion_aa",
    "ODesign-Rigid": "data/ood/odesign_rigid",
    "EvoDiff": "data/ood/evodiff",
    "GPDL": "data/ood/gpdl",
}

CHECKPOINTS = {
    "hybrid": "checkpoints/best_model.pth",
    "mlp": "checkpoints/baseline_mlp_best.pth",
    "gps": "checkpoints/baseline_gps_best.pth",
}


def collect_pdbs(
    root: str, max_per_generator: int | None = None, pattern: str = "**/*.pdb"
) -> list[str]:
    files = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    files = [
        f
        for f in files
        if "__MACOSX" not in f and not os.path.basename(f).startswith("._")
    ]
    if max_per_generator is not None and len(files) > max_per_generator:
        # Равномерно по мотивам (round-robin), а не первые попавшиеся
        by_motif: dict[str, list[str]] = {}
        for f in files:
            by_motif.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
        picked: list[str] = []
        queues = list(by_motif.values())
        while len(picked) < max_per_generator and any(queues):
            for q in queues:
                if q and len(picked) < max_per_generator:
                    picked.append(q.pop(0))
        files = picked
    return files


def parse_all(files: list[str], group: str) -> list[dict]:
    records = []
    skipped = 0
    for f in tqdm(files, desc=f"parse {group}", leave=False):
        try:
            coords = parse_pdb_to_backbone(f)
        except Exception:
            skipped += 1
            continue
        records.append(
            {
                "name": os.path.basename(f),
                "motif": os.path.basename(os.path.dirname(f)),
                "coords": center_coords(coords).astype(np.float32),
            }
        )
    if skipped:
        print(f"  {group}: пропущено {skipped} файлов")
    return records


def load_reference(manifest: str, n_each: int = 256) -> tuple[list[dict], list[dict]]:
    """Нативы и декoi из нашего test-сплита — референсные точки шкалы."""
    loader = make_loader(
        manifest, "test", batch_size=64, num_workers=0, shuffle=False, pin_memory=False
    )
    natives, decoys = [], []
    for batch in loader:
        labels = batch["label"].view(-1).numpy()
        for j in range(len(labels)):
            L = int(batch["length"][j])
            coords = batch["coords"][j].numpy()[:L].astype(np.float32)
            if labels[j] == 1 and len(natives) < n_each:
                natives.append({"name": f"native_{len(natives)}", "motif": "-", "coords": coords})
            elif labels[j] == 0 and len(decoys) < n_each:
                decoys.append({"name": f"decoy_{len(decoys)}", "motif": "-", "coords": coords})
        if len(natives) >= n_each and len(decoys) >= n_each:
            break
    return natives, decoys


@torch.no_grad()
def score_records(
    model, records: list[dict], device: torch.device, mc_runs: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Батчевый инференс; сортировка по длине минимизирует паддинг."""
    order = sorted(range(len(records)), key=lambda i: len(records[i]["coords"]))
    p_out = np.zeros(len(records), dtype=np.float32)
    u_out = np.zeros(len(records), dtype=np.float32)

    for start in tqdm(range(0, len(order), batch_size), desc="score", leave=False):
        idxs = order[start : start + batch_size]
        Lmax = max(len(records[i]["coords"]) for i in idxs)
        B = len(idxs)
        coords = np.zeros((B, Lmax, 3, 3), dtype=np.float32)
        mask = np.zeros((B, Lmax), dtype=bool)
        for b, i in enumerate(idxs):
            L = len(records[i]["coords"])
            coords[b, :L] = records[i]["coords"]
            mask[b, :L] = True

        ct = torch.from_numpy(coords).to(device)
        mt = torch.from_numpy(mask).to(device)
        with _autocast_ctx(device):
            res = predict_with_uncertainty(model, ct, mt, mc_runs=mc_runs)
        p_out[idxs] = res["p_foldable"].float().cpu().numpy()
        u_out[idxs] = res["uncertainty"].float().cpu().numpy()

    return p_out, u_out


def summarize(group: str, p: np.ndarray, u: np.ndarray) -> dict:
    return {
        "group": group,
        "n": int(len(p)),
        "p_fold_mean": float(p.mean()) if len(p) else float("nan"),
        "p_fold_median": float(np.median(p)) if len(p) else float("nan"),
        "frac_above_0.8": float((p > 0.8).mean()) if len(p) else float("nan"),
        "frac_above_0.5": float((p > 0.5).mean()) if len(p) else float("nan"),
        "frac_below_0.4": float((p <= 0.4).mean()) if len(p) else float("nan"),
        "uncertainty_mean": float(u.mean()) if len(p) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="OOD-оценка на выходах генераторов")
    parser.add_argument(
        "--dirs",
        action="append",
        default=None,
        help="Имя=путь к директории генератора (можно несколько). "
        "Если не задано — стандартные MotifBench-директории.",
    )
    parser.add_argument("--pattern", default="**/*.pdb", help="glob-паттерн структур внутри директории")
    parser.add_argument("--max-per-generator", type=int, default=None)
    parser.add_argument("--mc-runs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-reference", type=int, default=256)
    parser.add_argument("--manifest", default="dataset/manifest_v1_split.csv")
    parser.add_argument("--pca", default="dataset/pca_components.pth")
    parser.add_argument("-o", "--output", default="eval_results_generated.json")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.dirs:
        generator_dirs = {}
        for spec in args.dirs:
            name, _, path = spec.partition("=")
            if not path:
                raise ValueError(f"--dirs ожидает формат Имя=путь, получено: {spec}")
            generator_dirs[name] = path
    else:
        generator_dirs = GENERATOR_DIRS

    device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Устройство: {device} | mc_runs={args.mc_runs} | batch={args.batch_size}")

    # 1. Сбор и парсинг структур генераторов (один раз на все модели)
    generator_records: dict[str, list[dict]] = {}
    for gen, root in generator_dirs.items():
        if not os.path.isdir(root):
            print(f"Пропуск {gen}: нет {root}")
            continue
        files = collect_pdbs(root, args.max_per_generator, pattern=args.pattern)
        generator_records[gen] = parse_all(files, gen)
        print(f"{gen}: {len(generator_records[gen])} структур")

    natives, decoys = load_reference(args.manifest, n_each=args.n_reference)
    print(f"Референс: {len(natives)} нативов, {len(decoys)} декоев из test")

    # 2. Прогон каждой архитектуры
    all_results = {}
    detail_rows = []
    for arch, ckpt in CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            print(f"Пропуск {arch}: нет {ckpt}")
            continue
        print(f"\n===== {arch} ({ckpt}) =====")
        model, _ = build_eval_model(ckpt, device, pca_path=args.pca)

        summary = {}
        for gen, records in generator_records.items():
            p, u = score_records(model, records, device, args.mc_runs, args.batch_size)
            summary[gen] = summarize(gen, p, u)
            for rec, pi, ui in zip(records, p, u):
                detail_rows.append(
                    {
                        "model": arch,
                        "group": gen,
                        "motif": rec["motif"],
                        "name": rec["name"],
                        "length": len(rec["coords"]),
                        "p_fold": float(pi),
                        "uncertainty": float(ui),
                    }
                )
        p, u = score_records(model, natives, device, args.mc_runs, args.batch_size)
        summary["native_test"] = summarize("native_test", p, u)
        p, u = score_records(model, decoys, device, args.mc_runs, args.batch_size)
        summary["decoy_test"] = summarize("decoy_test", p, u)

        all_results[arch] = summary

    # 3. Сохранение
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(all_results, fp, ensure_ascii=False, indent=2)
    print(f"\nАгрегаты сохранены в {os.path.abspath(args.output)}")

    detail_path = os.path.splitext(args.output)[0] + "_detail.csv"
    with open(detail_path, "w", encoding="utf-8") as fp:
        fp.write("model,group,motif,name,length,p_fold,uncertainty\n")
        for r in detail_rows:
            fp.write(
                f"{r['model']},{r['group']},{r['motif']},{r['name']},"
                f"{r['length']},{r['p_fold']:.4f},{r['uncertainty']:.6f}\n"
            )
    print(f"Построчные результаты: {os.path.abspath(detail_path)}")

    # 4. Markdown-сводка
    groups = list(generator_dirs.keys()) + ["native_test", "decoy_test"]
    for arch, summary in all_results.items():
        print(f"\n=== {arch} ===")
        print("| Группа | n | P(fold) mean | median | >0.8 | >0.5 | <0.4 | uncert |")
        print("|---|---|---|---|---|---|---|---|")
        for g in groups:
            if g not in summary:
                continue
            s = summary[g]
            print(
                f"| {g} | {s['n']} | {s['p_fold_mean']:.3f} | {s['p_fold_median']:.3f} "
                f"| {s['frac_above_0.8']:.3f} | {s['frac_above_0.5']:.3f} "
                f"| {s['frac_below_0.4']:.3f} | {s['uncertainty_mean']:.5f} |"
            )


if __name__ == "__main__":
    main()
