"""
train_soft.py
-------------
Дообучение на мягкую метку: голова fold переводится с бинарной классификации
(BCE + сигмоида) на регрессию y = log1p(scRMSD) по self-consistency оракулу
MotifBench. Скор модели = предсказанный scRMSD (меньше = дизайнируемее),
без сигмоиды и MC-усреднения — ранжирование детерминированное.

Протокол hold-out:
  обучение — RFdiffusion, RFdiffusion-AA, EvoDiff (90% скаффолдов каждого);
  выбор эпохи — 10% скаффолдов обучающих генераторов (val);
  финальный отчёт — невиданные ODesign-Rigid и GPDL (см. analysis_enrichment.py).

Запуск:  python train_soft.py
"""

import glob
import os

import numpy as np
import torch
import torch.nn as nn

from analysis_logits_ranking import GENERATOR_DIRS, parse_pdb_files
from analysis_scrmsd import SCRMSD_AGG
from inference import build_model

TRAIN_GENERATORS = ["RFdiffusion", "RFdiffusion-AA", "EvoDiff"]
HOLDOUT_GENERATORS = ["ODesign-Rigid", "GPDL"]
EVAL_ROOTS = {
    "RFdiffusion": "data/ood/eval_rfdiffusion",
    "RFdiffusion-AA": "data/ood/eval_rfdiffusion_aa",
    "ODesign-Rigid": "data/ood/eval_odesign_rigid",
    "GPDL": "data/ood/eval_gpdl",
    "EvoDiff": "data/ood/eval_evodiff",
}

CKPT = "checkpoints/best_model.pth"
PCA = "dataset/pca_components.pth"
OUT = "checkpoints/soft_model.pth"

EPOCHS = 20
LR = 5e-5
BATCH = 32
VAL_FRACTION = 0.1
SEED = 42


def parse_scrmsd(root: str, agg: str = SCRMSD_AGG) -> dict:
    """(motif, sample) -> scRMSD скаффолда, агрегированный по 8 рефолдам."""
    import pandas as pd

    out = {}
    for f in glob.glob(os.path.join(root, "*", "*", "*", "self_consistency", "esm_eval_results.csv")):
        parts = f.split(os.sep)
        v = pd.to_numeric(pd.read_csv(f)["rmsd"], errors="coerce").dropna()
        if len(v) == 0:
            continue
        out[(parts[-4], parts[-3])] = float(v.min() if agg == "min" else v.mean())
    return out


def collect(group: str) -> tuple[list[dict], np.ndarray]:
    records, _ = parse_pdb_files(GENERATOR_DIRS[group])
    scrmsd = parse_scrmsd(EVAL_ROOTS[group])
    kept, ys = [], []
    for r in records:
        y = scrmsd.get((r["motif"], r["sample"].removesuffix(".pdb")))
        if y is not None:
            kept.append(r)
            ys.append(np.log1p(y))
    return kept, np.array(ys, dtype=np.float32)


def make_batches(records: list[dict], ys: np.ndarray, device: torch.device, shuffle: bool):
    from inference import center_coords

    n = len(records)
    order = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, BATCH):
        idxs = order[start : start + BATCH]
        Lmax = max(len(records[i]["coords"]) for i in idxs)
        B = len(idxs)
        coords = np.zeros((B, Lmax, 3, 3), dtype=np.float32)
        mask = np.zeros((B, Lmax), dtype=np.bool_)
        for b, i in enumerate(idxs):
            L = len(records[i]["coords"])
            coords[b, :L] = center_coords(records[i]["coords"])
            mask[b, :L] = True
        yield (
            torch.from_numpy(coords).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(ys[idxs]).to(device),
        )


@torch.no_grad()
def evaluate(model, records: list[dict], ys: np.ndarray, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0
    for coords, mask, y in make_batches(records, ys, device, shuffle=False):
        pred = model(coords, mask)["fold_logit"].float()
        total += loss_fn(pred, y).item() * len(y)
        n += len(y)
    model.train()
    return total / max(1, n)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    model = build_model(CKPT, device, pca_path=PCA, pair_init="scaled")

    # Данные: обучающие генераторы + train/val разбиение по скаффолдам
    train_records, train_ys = [], []
    val_records, val_ys = [], []
    for group in TRAIN_GENERATORS:
        records, ys = collect(group)
        n_val = int(len(records) * VAL_FRACTION)
        perm = np.random.permutation(len(records))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        val_records += [records[i] for i in val_idx]
        val_ys.append(ys[val_idx])
        train_records += [records[i] for i in train_idx]
        train_ys.append(ys[train_idx])
        print(f"{group}: {len(records)} размечено, val={n_val}, train={len(train_idx)}")
    train_ys = np.concatenate(train_ys)
    val_ys = np.concatenate(val_ys)

    holdout = {}
    for group in HOLDOUT_GENERATORS:
        records, ys = collect(group)
        holdout[group] = (records, ys)
        print(f"{group} (holdout): {len(records)} размечено")

    # Дообучение: голова fold — регрессия log1p(scRMSD), без сигмоиды
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    print(f"\nДообучение: {EPOCHS} эпох, lr={LR}, batch={BATCH}, train={len(train_records)}")
    print(f"{'эпоха':>5} | {'train MSE':>9} | {'val MSE':>8} | {'holdout ODesign':>15} | {'holdout GPDL':>12}")
    best_val, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        total, n = 0.0, 0
        for coords, mask, y in make_batches(train_records, train_ys, device, shuffle=True):
            opt.zero_grad()
            pred = model(coords, mask)["fold_logit"].float()
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        train_mse = total / n
        val_mse = evaluate(model, val_records, val_ys, device)
        h_odesign = evaluate(model, *holdout["ODesign-Rigid"], device)
        h_gpdl = evaluate(model, *holdout["GPDL"], device)
        print(f"{epoch:>5} | {train_mse:>9.4f} | {val_mse:>8.4f} | {h_odesign:>15.4f} | {h_gpdl:>12.4f}")
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    torch.save(
        {"model_state_dict": best_state, "source": CKPT, "target": "log1p(scRMSD)", "val_mse": best_val},
        OUT,
    )
    print(f"\nЛучшая эпоха по val MSE={best_val:.4f}; сохранено: {os.path.abspath(OUT)}")
    print("holdout-MSE лучшей эпохи пересчитает analysis_enrichment.py (обогащение, не MSE)")


if __name__ == "__main__":
    main()
