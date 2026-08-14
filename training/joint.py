"""
training/joint.py
--------------
Совместное обучение: по-остаточный lDDT + глобальная мягкая метка log1p(scRMSD).

Зачем именно так (результат training/perresidue.py): чистая по-остаточная
супервизия подняла внутримотивное ранжирование там, где локальное окружение
и глобальное RMSD согласованы (RFdiffusion 0.32 → 0.51, холдаут ODesign
0.30 → 0.38), но просела на EvoDiff, где эта связь слабая: внутримотивный
Spearman(mean lDDT, scRMSD) там всего −0.48 против −0.86…−0.94 у остальных.
То есть модель оптимизировала не то, чем её меряют.

Совместный лосс должен дать и то, и другое: плотную супервизию (~1.2M меток
против 8.1k структур) для тонкой дискриминации + прямой глобальный таргет,
сохраняющий динамический диапазон там, где lDDT расходится с scRMSD.

Два выхода для ранжирования, сравниваются в analysis/perresidue.py:
  * fold_logit → expm1 — предсказанный scRMSD (прямая супервизия);
  * mean sigmoid(lddt_logit) — агрегат по-остаточного предсказания.

Данные те же, что у training/soft.py и training/perresidue.py (старт из
best_model.pth, обучение на RFdiffusion + RFdiffusion-AA + EvoDiff, 90/10,
холдаут ODesign-Rigid и GPDL). Оптимизация — НЕ та же: здесь AdamW с
weight_decay 1e-4, клиппингом нормы 1.0, bf16-автокастом и 45 эпохами, тогда
как training/soft.py учится Adam'ом без регуляризации и клиппинга, в fp32 и
за 20 эпох. Поэтому сравнение «joint против soft» смешивает разрешение метки
с режимом оптимизации; чтобы приписать разницу супервизии, обе стадии нужно
переобучить под одним протоколом (docs/03).

Запуск:  python -m training.joint
"""

import numpy as np
import torch
import torch.nn.functional as F

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
from training.perresidue import load_labels, masked_bce
from training.soft import PCA

CKPT = "checkpoints/best_model.pth"
OUT = "checkpoints/joint_model.pth"

EPOCHS = 45
LR = 5e-5
BATCH = 32
SEED = 42
W_RESIDUE = 1.0
W_GLOBAL = 1.0


def collect(generator: str) -> list[dict]:
    """Структуры, у которых есть ОБЕ метки: по-остаточная и глобальная."""
    records, _ = parse_pdb_files(GENERATOR_DIRS[generator])
    lddt = load_labels(generator)
    scrmsd = scrmsd_by_scaffold(EVAL_SOURCES[generator])
    kept, skipped = [], 0
    for r in records:
        sample = r["sample"].removesuffix(".pdb")
        y_res = lddt.get(f"{r['motif']}/{sample}")
        y_glob = scrmsd.get((r["motif"], sample))
        if y_res is None or y_glob is None or len(y_res) != len(r["coords"]):
            skipped += 1
            continue
        kept.append({**r, "lddt": y_res, "log_scrmsd": np.float32(np.log1p(y_glob))})
    print(f"  {generator:16} структур с обеими метками: {len(kept):5d} (пропущено {skipped})")
    return kept


def make_batches(records: list[dict], device: torch.device, shuffle: bool):
    """Батчи (coords, mask, по-остаточный таргет, глобальный таргет)."""
    order = "shuffle" if shuffle else "sequential"
    for idxs, coords, mask in iter_padded_batches(records, BATCH, device, order=order):
        t_res = np.zeros(tuple(mask.shape), dtype=np.float32)
        t_glob = np.zeros((len(idxs),), dtype=np.float32)
        for b, i in enumerate(idxs):
            t_res[b, : len(records[i]["lddt"])] = records[i]["lddt"]
            t_glob[b] = records[i]["log_scrmsd"]
        yield (
            coords,
            mask,
            torch.from_numpy(t_res).to(device),
            torch.from_numpy(t_glob).to(device),
        )


@torch.no_grad()
def evaluate(model, records, device) -> tuple[float, float, float]:
    model.eval()
    res_sum, glob_sum, n_struct = 0.0, 0.0, 0
    for coords, mask, t_res, t_glob in make_batches(records, device, shuffle=False):
        with _autocast_ctx(device):
            out = model(coords, mask)
        lddt_logit = out["lddt_logit"].float()
        fold = out["fold_logit"].float()
        res_sum += masked_bce(lddt_logit, t_res, mask).item() * len(t_glob)
        glob_sum += F.mse_loss(fold, t_glob, reduction="sum").item()
        n_struct += len(t_glob)
    model.train()
    res_loss = res_sum / max(1, n_struct)
    glob_loss = glob_sum / max(1, n_struct)
    return res_loss, glob_loss, W_RESIDUE * res_loss + W_GLOBAL * glob_loss


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device} | веса лосса: остатки {W_RESIDUE}, глобальный {W_GLOBAL}")

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
    print(f"train {len(train_records)} структур / {n_labels} по-остаточных меток "
          f"+ {len(train_records)} глобальных, val {len(val_records)}")
    print(f"Холдаут: {', '.join(HOLDOUT_GENERATORS)}")

    model = build_model(CKPT, device, pca_path=PCA, per_residue=True,
                        pair_init="scaled")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        run_res, run_glob, nb = 0.0, 0.0, 0
        for coords, mask, t_res, t_glob in make_batches(train_records, device, shuffle=True):
            with _autocast_ctx(device):
                out = model(coords, mask)
            l_res = masked_bce(out["lddt_logit"].float(), t_res, mask)
            l_glob = F.mse_loss(out["fold_logit"].float(), t_glob)
            loss = W_RESIDUE * l_res + W_GLOBAL * l_glob
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_res += l_res.item()
            run_glob += l_glob.item()
            nb += 1

        v_res, v_glob, v_total = evaluate(model, val_records, device)
        flag = ""
        if v_total < best:
            best = v_total
            torch.save({"model_state_dict": model.state_dict()}, OUT)
            flag = " ← сохранено"
        print(f"эпоха {epoch:2d} | train: res {run_res / nb:.4f} glob {run_glob / nb:.4f} | "
              f"val: res {v_res:.4f} glob {v_glob:.4f} сумма {v_total:.4f}{flag}", flush=True)

    print(f"\nЛучший чекпоинт: {OUT} (val сумма {best:.4f})")


if __name__ == "__main__":
    main()
