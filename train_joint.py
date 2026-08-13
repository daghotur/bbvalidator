"""
train_joint.py
--------------
Совместное обучение: по-остаточный lDDT + глобальная мягкая метка log1p(scRMSD).

Зачем именно так (результат train_perresidue.py): чистая по-остаточная
супервизия подняла внутримотивное ранжирование там, где локальное окружение
и глобальное RMSD согласованы (RFdiffusion 0.32 → 0.51, холдаут ODesign
0.30 → 0.38), но просела на EvoDiff, где эта связь слабая: внутримотивный
Spearman(mean lDDT, scRMSD) там всего −0.48 против −0.86…−0.94 у остальных.
То есть модель оптимизировала не то, чем её меряют.

Совместный лосс должен дать и то, и другое: плотную супервизию (~1.2M меток
против 8.1k структур) для тонкой дискриминации + прямой глобальный таргет,
сохраняющий динамический диапазон там, где lDDT расходится с scRMSD.

Два выхода для ранжирования, сравниваются в analysis_perresidue.py:
  * fold_logit → expm1 — предсказанный scRMSD (прямая супервизия);
  * mean sigmoid(lddt_logit) — агрегат по-остаточного предсказания.

Протокол тот же, что у train_soft.py и train_perresidue.py (старт из
best_model.pth, обучение на RFdiffusion + RFdiffusion-AA + EvoDiff, 90/10,
холдаут ODesign-Rigid и GPDL) — чтобы разница была в супервизии, не в данных.

Запуск:  python train_joint.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from analysis_logits_ranking import GENERATOR_DIRS, parse_pdb_files
from inference import _autocast_ctx, build_model, center_coords
from train_perresidue import load_labels, masked_bce
from train_soft import EVAL_ROOTS, HOLDOUT_GENERATORS, PCA, TRAIN_GENERATORS, parse_scrmsd

CKPT = "checkpoints/best_model.pth"
OUT = "checkpoints/joint_model.pth"

EPOCHS = 45
LR = 5e-5
BATCH = 32
VAL_FRACTION = 0.1
SEED = 42
W_RESIDUE = 1.0
W_GLOBAL = 1.0


def collect(generator: str) -> list[dict]:
    """Структуры, у которых есть ОБЕ метки: по-остаточная и глобальная."""
    records, _ = parse_pdb_files(GENERATOR_DIRS[generator])
    lddt = load_labels(generator)
    scrmsd = parse_scrmsd(EVAL_ROOTS[generator])
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
    n = len(records)
    order = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, BATCH):
        idxs = order[start : start + BATCH]
        Lmax = max(len(records[i]["coords"]) for i in idxs)
        B = len(idxs)
        coords = np.zeros((B, Lmax, 3, 3), dtype=np.float32)
        mask = np.zeros((B, Lmax), dtype=np.bool_)
        t_res = np.zeros((B, Lmax), dtype=np.float32)
        t_glob = np.zeros((B,), dtype=np.float32)
        for b, i in enumerate(idxs):
            L = len(records[i]["coords"])
            coords[b, :L] = center_coords(records[i]["coords"])
            mask[b, :L] = True
            t_res[b, :L] = records[i]["lddt"]
            t_glob[b] = records[i]["log_scrmsd"]
        yield (
            torch.from_numpy(coords).to(device),
            torch.from_numpy(mask).to(device),
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
    train_records, val_records = [], []
    rng = np.random.default_rng(SEED)
    for gen in TRAIN_GENERATORS:
        recs = collect(gen)
        idx = rng.permutation(len(recs))
        n_val = int(len(recs) * VAL_FRACTION)
        val_records += [recs[i] for i in idx[:n_val]]
        train_records += [recs[i] for i in idx[n_val:]]

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
