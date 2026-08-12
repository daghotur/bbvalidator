"""
train_soft_multitask.py
-----------------------
Починка вспомогательных голов после однозадачного soft-дообучения:
энкодер сдвинулся к ранжирующему представлению, а rmsd/steric/failure_mode
головы не получали градиент и отстали. Мультизадачное дообучение из
soft_model.pth двумя потоками:

  A — датасет (manifest_v1_split, train): лоссы вспомогательных голов
      (MSE log1p-RMSD, MSE стерика, CE failure_mode с весами классов);
  B — структуры генераторов с scRMSD: MSE мягкой метки в голову fold
      (та же цель, что в train_soft.py).

Fold-голова BCE-лосс не получает: она теперь регрессионная. Потоки идут с
разными lr (вспомогательные головы выше, остальная сеть ниже), чтобы
восстановить головы, не размывая ранжирующий сигнал.

Третий якорь — нативы: позитивы из aux-батчей получают мягкую метку
log1p(1 Å) (нативная последовательность дизайнуема по построению). Без этого
якоря fold-голова никогда не видит нативов и предсказывает им мусорные
значения (слепая зона чистого soft-дообучения).

Запуск:  python train_soft_multitask.py
"""

import itertools

import numpy as np
import torch
import torch.nn.functional as F

from dataset.dataloader import get_dataloaders
from inference import build_model
from train_model import MANIFEST_PATH, _autocast_ctx, load_failure_class_weights
from train_soft import (
    BATCH,
    HOLDOUT_GENERATORS,
    OUT as SOFT_CKPT,
    PCA,
    TRAIN_GENERATORS,
    collect,
    make_batches,
)

OUT = "checkpoints/soft_model_mt.pth"
EPOCHS = 3
STEPS_PER_EPOCH = 300      # пар батчей (aux + soft) за эпоху
LR_MAIN = 1e-5             # энкодер/пулер/fold-голова
LR_AUX = 3e-4              # вспомогательные головы
AUX_BATCH = 64
SEED = 42
VAL_FRACTION = 0.1
NATIVE_SCRMSD_TARGET = 1.0  # Å: натив дизайнуем по построению
W_NATIVE = 0.25             # вес native-якоря: полный вес ломал ранжирование генераторов


@torch.no_grad()
def eval_soft(model, records, ys, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for coords, mask, y in make_batches(records, ys, device, shuffle=False):
        with _autocast_ctx(device):
            pred = model(coords, mask)["fold_logit"].float()
        total += F.mse_loss(pred, y, reduction="sum").item()
        n += len(y)
    model.train()
    return total / max(1, n)


@torch.no_grad()
def eval_native_pred(model, loader, device, max_batches=100) -> float:
    """Средний предсказанный scRMSD (Å) на нативах val-сплита."""
    model.eval()
    preds_all = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        pos = batch["label"].view(-1) == 1
        if not pos.any():
            continue
        coords = batch["coords"].to(device)[pos]
        mask = batch["mask"].to(device)[pos]
        with _autocast_ctx(device):
            preds_all.append(torch.expm1(model(coords, mask)["fold_logit"].float()).cpu())
    model.train()
    all_p = torch.cat(preds_all)
    return float(all_p.mean())


@torch.no_grad()
def eval_aux(model, loader, class_weights, device, max_batches=100) -> dict:
    model.eval()
    se_rmsd = se_steric = correct = total = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        coords = batch["coords"].to(device)
        mask = batch["mask"].to(device)
        with _autocast_ctx(device):
            preds = model(coords, mask)
        se_rmsd += F.mse_loss(
            preds["rmsd"].float(),
            torch.log1p(batch["rmsd_target"].float().to(device)),
            reduction="sum",
        ).item()
        se_steric += F.mse_loss(
            preds["steric"].float(), batch["steric_target"].float().to(device),
            reduction="sum",
        ).item()
        fm = preds["failure_mode"].argmax(-1).cpu()
        correct += (fm == batch["failure_mode_label"]).sum().item()
        total += len(fm)
    model.train()
    return {"rmsd_mse": se_rmsd / total, "steric_mse": se_steric / total,
            "fail_acc": correct / total}


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    model = build_model(SOFT_CKPT, device, pca_path=PCA)
    class_weights = load_failure_class_weights(MANIFEST_PATH).to(device)

    # Поток A: датасет для вспомогательных голов
    print(f"Загрузка данных из {MANIFEST_PATH}...")
    train_loader, val_loader, _ = get_dataloaders(
        MANIFEST_PATH, batch_size=AUX_BATCH, num_workers=8, pin_memory=True, seed=SEED
    )

    # Поток B: мягкая метка по структурам генераторов
    train_records, train_ys, val_records, val_ys = [], [], [], []
    for group in TRAIN_GENERATORS:
        records, ys = collect(group)
        n_val = int(len(records) * VAL_FRACTION)
        perm = np.random.permutation(len(records))
        val_records += [records[i] for i in perm[:n_val]]
        val_ys.append(ys[perm[:n_val]])
        train_records += [records[i] for i in perm[n_val:]]
        train_ys.append(ys[perm[n_val:]])
    train_ys = np.concatenate(train_ys)
    val_ys = np.concatenate(val_ys)
    holdout = {g: collect(g) for g in HOLDOUT_GENERATORS}
    print(f"Soft-поток: train={len(train_records)}, val={len(val_records)}, "
          f"holdout={ {g: len(r) for g, (r, _) in holdout.items()} }")

    # Оптимизатор: вспомогательные головы учатся быстрее остальной сети
    aux_params = [
        p
        for name, head in [
            ("rmsd", model.heads.rmsd_head),
            ("steric", model.heads.steric_head),
            ("failure_mode", model.heads.failure_mode_head),
        ]
        for p in head.parameters()
    ]
    aux_ids = {id(p) for p in aux_params}
    main_params = [p for p in model.parameters() if p.requires_grad and id(p) not in aux_ids]
    opt = torch.optim.AdamW(
        [
            {"params": main_params, "lr": LR_MAIN},
            {"params": aux_params, "lr": LR_AUX},
        ],
        weight_decay=1e-4,
    )

    # Точка отсчёта до дообучения
    soft_val0 = eval_soft(model, val_records, val_ys, device)
    aux0 = eval_aux(model, val_loader, class_weights, device)
    native0 = eval_native_pred(model, val_loader, device)
    print(f"До дообучения: soft val MSE={soft_val0:.4f} | aux: {aux0} | "
          f"предсказание на нативах {native0:.2f} Å (цель {NATIVE_SCRMSD_TARGET:.1f})")

    # Сохраняется финальная эпоха: native-якорь важнее минимума val-MSE,
    # а дрейф за 3 эпохи контролируется по holdout в логе.
    model.train()
    for epoch in range(1, EPOCHS + 1):
        train_loader.batch_sampler.set_epoch(epoch)
        aux_it = itertools.cycle(train_loader)
        soft_it = itertools.cycle(make_batches(train_records, train_ys, device, shuffle=True))

        sums = {"soft": 0.0, "rmsd": 0.0, "steric": 0.0, "fail": 0.0, "native": 0.0}
        for _ in range(STEPS_PER_EPOCH):
            batch = next(aux_it)
            coords_a = batch["coords"].to(device, non_blocking=True)
            mask_a = batch["mask"].to(device, non_blocking=True)
            coords_s, mask_s, y_s = next(soft_it)

            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(device):
                pa = model(coords_a, mask_a)
                l_rmsd = F.mse_loss(pa["rmsd"].float(), torch.log1p(batch["rmsd_target"].float().to(device)))
                l_steric = F.mse_loss(pa["steric"].float(), batch["steric_target"].float().to(device))
                l_fail = F.cross_entropy(
                    pa["failure_mode"].float(),
                    batch["failure_mode_label"].long().to(device),
                    weight=class_weights,
                )
                ps = model(coords_s, mask_s)
                l_soft = F.mse_loss(ps["fold_logit"].float(), y_s)
                pos = batch["label"].view(-1) == 1
                if pos.any():
                    native_t = torch.full(
                        (int(pos.sum()),), float(np.log1p(NATIVE_SCRMSD_TARGET)), device=device
                    )
                    l_native = W_NATIVE * F.mse_loss(pa["fold_logit"].float()[pos], native_t)
                else:
                    l_native = torch.zeros((), device=device)
                loss = l_soft + l_rmsd + l_steric + l_fail + l_native
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            sums["soft"] += l_soft.item()
            sums["rmsd"] += l_rmsd.item()
            sums["steric"] += l_steric.item()
            sums["fail"] += l_fail.item()
            sums["native"] += l_native.item()

        aux_val = eval_aux(model, val_loader, class_weights, device)
        soft_val = eval_soft(model, val_records, val_ys, device)
        native_pred = eval_native_pred(model, val_loader, device)
        h_odesign = eval_soft(model, *holdout["ODesign-Rigid"], device)
        h_gpdl = eval_soft(model, *holdout["GPDL"], device)
        avg = {k: v / STEPS_PER_EPOCH for k, v in sums.items()}
        print(
            f"Эпоха {epoch}: soft={avg['soft']:.4f} native={avg['native']:.4f} "
            f"rmsd={avg['rmsd']:.4f} steric={avg['steric']:.4f} fail={avg['fail']:.4f} | "
            f"val soft MSE {soft_val:.4f} (было {soft_val0:.4f}) | "
            f"нативы {native_pred:.2f} Å (было {native0:.2f}) | "
            f"aux val: rmsd {aux_val['rmsd_mse']:.4f}, fail_acc {aux_val['fail_acc']:.3f} | "
            f"holdout ODesign {h_odesign:.4f}, GPDL {h_gpdl:.4f}"
        )
    final_val = eval_soft(model, val_records, val_ys, device)
    torch.save(
        {"model_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
         "source": SOFT_CKPT,
         "recipe": "multitask aux repair + soft MSE + native anchor",
         "val_mse": final_val},
        OUT,
    )
    print(f"\nСохранено: {OUT} (val soft MSE {final_val:.4f})")
    print("Финальная проверка: analysis_enrichment.py (ранжирование) и eval_model.py (aux-головы)")


if __name__ == "__main__":
    main()
