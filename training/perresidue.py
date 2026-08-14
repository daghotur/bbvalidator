"""
training/perresidue.py
-------------------
Дообучение на ПО-ОСТАТОЧНУЮ метку вместо одного скаляра на структуру.

Мотивация (измерено, см. analysis/oracle_ceiling.py): метка scRMSD надёжна
(R = 0.85-0.96), но модель забирает лишь 32-54% доступного внутримотивного
сигнала. Узкое место — не сложность задачи и не шум оракула, а плотность
супервизии: ~15k структур × 1 скаляр. По-остаточный таргет даёт ~1.5M меток
из тех же данных, без единого нового прогона ESMFold.

Таргет — CA-lDDT против ESMFold-рефолда, усреднённый по 8 последовательностям
(training/build_lddt_labels.py). Лосс — BCE по мягкой метке, маскированный по валидным
остаткам. Глобальный скор для ранжирования выводится агрегацией:
score = mean_i sigmoid(lddt_logit_i), больше = дизайнируемее.

Протокол сравнения с training/soft.py — тот же, чтобы разница была в супервизии,
а не в данных: старт из checkpoints/best_model.pth, обучение на RFdiffusion +
RFdiffusion-AA + EvoDiff (90% скаффолдов), выбор эпохи по 10% их же скаффолдов,
ODesign-Rigid и GPDL целиком в холдауте.

Запуск:  python -m training.perresidue
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from common.motifbench import (
    GENERATOR_DIRS,
    HOLDOUT_GENERATORS,
    TRAIN_GENERATORS,
    motif_split,
)
from common.structures import iter_padded_batches, parse_pdb_files
from inference import _autocast_ctx, build_model
from training.soft import PCA

CKPT = "checkpoints/best_model.pth"
OUT = "checkpoints/perres_model.pth"
LABEL_DIR = "dataset/lddt_labels"

EPOCHS = 20
LR = 5e-5
BATCH = 32
SEED = 42


def load_labels(generator: str) -> dict:
    path = os.path.join(LABEL_DIR, f"{generator.replace('/', '_')}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} — сначала запустите python -m training.build_lddt_labels")
    with np.load(path) as z:
        # [2, L]: строка 0 — среднее lDDT по рефолдам, строка 1 — std (согласие оракула)
        return {k: z[k][0].astype(np.float32) for k in z.files}


def collect(generator: str) -> list[dict]:
    """Скаффолды с по-остаточной меткой той же длины."""
    records, _ = parse_pdb_files(GENERATOR_DIRS[generator])
    labels = load_labels(generator)
    kept, mismatched = [], 0
    for r in records:
        key = f"{r['motif']}/{r['sample'].removesuffix('.pdb')}"
        y = labels.get(key)
        if y is None:
            continue
        if len(y) != len(r["coords"]):
            mismatched += 1
            continue
        kept.append({**r, "lddt": y})
    print(f"  {generator:16} структур с меткой: {len(kept):5d} "
          f"(длина не совпала: {mismatched})")
    return kept


def make_batches(records: list[dict], device: torch.device, shuffle: bool):
    """Батчи (coords, mask, по-остаточный таргет lDDT)."""
    order = "shuffle" if shuffle else "sequential"
    for idxs, coords, mask in iter_padded_batches(records, BATCH, device, order=order):
        target = np.zeros(tuple(mask.shape), dtype=np.float32)
        for b, i in enumerate(idxs):
            target[b, : len(records[i]["lddt"])] = records[i]["lddt"]
        yield coords, mask, torch.from_numpy(target).to(device)


def masked_bce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, records, device) -> tuple[float, float]:
    """Возвращает (BCE, MAE по остаткам)."""
    model.eval()
    total_bce, total_mae, n_res = 0.0, 0.0, 0
    for coords, mask, target in make_batches(records, device, shuffle=False):
        with _autocast_ctx(device):
            logit = model(coords, mask)["lddt_logit"].float()
        m = mask.float()
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        mae = (torch.sigmoid(logit) - target).abs()
        total_bce += (bce * m).sum().item()
        total_mae += (mae * m).sum().item()
        n_res += m.sum().item()
    model.train()
    return total_bce / max(1, n_res), total_mae / max(1, n_res)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    print("Сбор обучающих генераторов:")
    collected = {gen: collect(gen) for gen in TRAIN_GENERATORS}
    train_motifs, val_motifs = motif_split(
        {r["motif"] for recs in collected.values() for r in recs}
    )
    print(f"Мотивов: {len(train_motifs)} в обучении, {len(val_motifs)} отложено "
          f"под выбор эпохи ({', '.join(sorted(val_motifs))})")

    train_records, val_records = [], []
    for gen, recs in collected.items():
        val_records += [r for r in recs if r["motif"] in val_motifs]
        train_records += [r for r in recs if r["motif"] not in val_motifs]

    n_labels = sum(len(r["lddt"]) for r in train_records)
    print(f"train {len(train_records)} структур / {n_labels} по-остаточных меток, "
          f"val {len(val_records)} структур")
    print(f"Холдаут (не участвует в обучении): {', '.join(HOLDOUT_GENERATORS)}")

    model = build_model(CKPT, device, pca_path=PCA, per_residue=True,
                        pair_init="scaled")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        run_loss, n_batches = 0.0, 0
        for coords, mask, target in make_batches(train_records, device, shuffle=True):
            with _autocast_ctx(device):
                logit = model(coords, mask)["lddt_logit"].float()
            loss = masked_bce(logit, target, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()
            n_batches += 1

        val_bce, val_mae = evaluate(model, val_records, device)
        flag = ""
        if val_bce < best:
            best = val_bce
            torch.save({"model_state_dict": model.state_dict()}, OUT)
            flag = " ← сохранено"
        print(f"эпоха {epoch:2d} | train BCE {run_loss / max(1, n_batches):.4f} | "
              f"val BCE {val_bce:.4f} | val MAE {val_mae:.4f}{flag}")

    print(f"\nЛучший чекпоинт: {OUT} (val BCE {best:.4f})")
    print("Дальше: python -m analysis.perresidue — within-motif ранжирование против soft_model")


if __name__ == "__main__":
    main()
