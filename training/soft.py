"""
training/soft.py
-------------
Дообучение на мягкую метку: голова fold переводится с бинарной классификации
(BCE + сигмоида) на регрессию y = log1p(scRMSD) по self-consistency оракулу
MotifBench. Скор модели = предсказанный scRMSD (меньше = дизайнируемее),
без сигмоиды и MC-усреднения — ранжирование детерминированное.

Протокол hold-out:
  обучение — RFdiffusion, RFdiffusion-AA, EvoDiff (90% скаффолдов каждого);
  выбор эпохи — 10% скаффолдов обучающих генераторов (val);
  финальный отчёт — невиданные ODesign-Rigid и GPDL (см. analysis/enrichment.py).

Запуск:  python -m training.soft
"""

import os

import numpy as np
import torch
import torch.nn as nn

from common.motifbench import (
    EVAL_SOURCES,
    GENERATOR_DIRS,
    HOLDOUT_GENERATORS,
    TRAIN_GENERATORS,
    motif_split,
    scrmsd_by_scaffold,
)
from common.structures import iter_padded_batches, parse_pdb_files
from inference import _autocast_ctx, build_model

CKPT = "checkpoints/best_model.pth"
PCA = "dataset/pca_components.pth"
OUT = "checkpoints/soft_model.pth"

EPOCHS = 20
LR = 5e-5
BATCH = 32
SEED = 42


def collect(group: str) -> tuple[list[dict], np.ndarray]:
    records, _ = parse_pdb_files(GENERATOR_DIRS[group])
    scrmsd = scrmsd_by_scaffold(EVAL_SOURCES[group])
    kept, ys = [], []
    for r in records:
        y = scrmsd.get((r["motif"], r["sample"].removesuffix(".pdb")))
        if y is not None:
            kept.append(r)
            ys.append(np.log1p(y))
    return kept, np.array(ys, dtype=np.float32)


def make_batches(records: list[dict], ys: np.ndarray, device: torch.device, shuffle: bool):
    """Батчи (coords, mask, таргет) в порядке случайном или как есть."""
    order = "shuffle" if shuffle else "sequential"
    for idxs, coords, mask in iter_padded_batches(records, BATCH, device, order=order):
        yield coords, mask, torch.from_numpy(ys[idxs]).to(device)


@torch.no_grad()
def evaluate(model, records: list[dict], ys: np.ndarray, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0
    for coords, mask, y in make_batches(records, ys, device, shuffle=False):
        with _autocast_ctx(device):
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

    # Данные: обучающие генераторы, val — целиком отложенные мотивы
    collected = {g: collect(g) for g in TRAIN_GENERATORS}
    all_motifs = {r["motif"] for records, _ in collected.values() for r in records}
    train_motifs, val_motifs = motif_split(all_motifs)
    print(f"Мотивов: {len(train_motifs)} в обучении, {len(val_motifs)} отложено "
          f"под выбор эпохи ({', '.join(sorted(val_motifs))})")

    train_records, train_ys = [], []
    val_records, val_ys = [], []
    for group, (records, ys) in collected.items():
        is_val = np.array([r["motif"] in val_motifs for r in records])
        val_records += [r for r, v in zip(records, is_val) if v]
        val_ys.append(ys[is_val])
        train_records += [r for r, v in zip(records, is_val) if not v]
        train_ys.append(ys[~is_val])
        print(f"{group}: {len(records)} размечено, val={int(is_val.sum())}, "
              f"train={int((~is_val).sum())}")
    train_ys = np.concatenate(train_ys)
    val_ys = np.concatenate(val_ys)

    holdout = {}
    for group in HOLDOUT_GENERATORS:
        records, ys = collect(group)
        holdout[group] = (records, ys)
        print(f"{group} (holdout): {len(records)} размечено")

    # Дообучение: голова fold — регрессия log1p(scRMSD), без сигмоиды
    loss_fn = nn.MSELoss()
    # Единый с training/perresidue.py и training/joint.py режим: AdamW с
    # weight decay, клиппинг нормы, bf16-автокаст. Иначе сравнение стадий
    # смешивает разрешение метки с режимом оптимизации.
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()

    print(f"\nДообучение: {EPOCHS} эпох, lr={LR}, batch={BATCH}, train={len(train_records)}")
    print(f"{'эпоха':>5} | {'train MSE':>9} | {'val MSE':>8} | {'holdout ODesign':>15} | {'holdout GPDL':>12}")
    best_val, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        total, n = 0.0, 0
        for coords, mask, y in make_batches(train_records, train_ys, device, shuffle=True):
            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(device):
                pred = model(coords, mask)["fold_logit"].float()
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    print("holdout-MSE лучшей эпохи пересчитает python -m analysis.enrichment (обогащение, не MSE)")


if __name__ == "__main__":
    main()
