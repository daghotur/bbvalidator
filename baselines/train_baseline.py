"""
baselines/train_baseline.py
---------------------------
Обучение базлайнов (MLP / GPS) по тому же протоколу, что и основная модель:
тот же фронтенд с PCA, тот же пулинг/головы/динамический лосс, те же сиды,
бакеты по длине и селекция по (Val AUC + PR-AUC)/2. Цикл обучения переиспользуется
из train_model.py — отличается только энкодер.

Запуск:
    python baselines/train_baseline.py --encoder mlp
    python baselines/train_baseline.py --encoder gps
"""

import argparse
import os
import sys

# Запуск из корня (python baselines/train_baseline.py) кладёт в sys.path
# каталог скрипта, а не репозиторий — добавляем корень явно.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.tensorboard import SummaryWriter
from preprocess.pair_features import PairFeatureBuilder

import train_model as tm
from baselines.encoders import BaselineGPSEncoder, BaselineMLPEncoder
from dataset.dataloader import get_dataloaders
from model.heads_loss import (
    DynamicMultiTaskLoss,
    MultiHeadAttentionPooling,
    ProteinMultiTaskHeads,
    ProteinScoreModel,
)
from preprocess.biophys_frontend import BiophysicalFrontend
from preprocess.fit_pca import load_pca_into_frontend

D_MODEL = 128  # базлайны меньше основного энкодера (d_model=192) — как в исходных main.py/main2.py
NUM_EPOCHS = 15


def build_baseline(encoder_name: str, device: torch.device) -> ProteinScoreModel:
    frontend = BiophysicalFrontend(use_no_grad=True)
    load_pca_into_frontend(frontend, tm.PCA_PATH)

    if encoder_name == "mlp":
        encoder = BaselineMLPEncoder(node_in_dim=31, d_model=D_MODEL, dropout=0.15)
    elif encoder_name == "gps":
        encoder = BaselineGPSEncoder(
            node_in_dim=31, pair_in_dim=PairFeatureBuilder().feature_dim,
            d_model=D_MODEL, heads=4,
            num_layers=2, dropout=0.15,
        )
    else:
        raise ValueError(f"Неизвестный базлайн: {encoder_name}")

    pooler = MultiHeadAttentionPooling(d_model=D_MODEL, num_heads=4)
    heads = ProteinMultiTaskHeads(d_model=D_MODEL, dropout=0.15)
    return ProteinScoreModel(frontend, encoder, pooler, heads).to(device)


def main():
    parser = argparse.ArgumentParser(description="Обучение базлайна (mlp|gps)")
    parser.add_argument("--encoder", choices=["mlp", "gps"], required=True)
    args = parser.parse_args()

    tm.set_seed(tm.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Базлайн '{args.encoder}' на {device} (seed={tm.SEED}, d_model={D_MODEL})")

    model = build_baseline(args.encoder, device)

    class_weights = tm.load_failure_class_weights(tm.MANIFEST_PATH)
    criterion = DynamicMultiTaskLoss(
        num_tasks=tm.NUM_TASKS, failure_class_weights=class_weights
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": trainable, "lr": 6e-4, "weight_decay": 1e-4},
        {"params": criterion.parameters(), "lr": 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )

    print(f"Загрузка данных из {tm.MANIFEST_PATH}...")
    train_loader, val_loader, _ = get_dataloaders(
        tm.MANIFEST_PATH, batch_size=64, num_workers=8, pin_memory=True, seed=tm.SEED
    )

    os.makedirs("checkpoints", exist_ok=True)
    run_name = f"Baseline_{args.encoder.upper()}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    best_composite = 0.0
    global_step = 0
    ckpt_best = f"checkpoints/baseline_{args.encoder}_best.pth"
    ckpt_last = f"checkpoints/baseline_{args.encoder}_last.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loader.batch_sampler.set_epoch(epoch)

        train_loss, train_acc, global_step = tm.train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, writer, global_step
        )
        val_loss, val_acc, val_auc, val_prauc, composite = tm.validate(
            model, criterion, val_loader, device, epoch, writer
        )
        scheduler.step()

        print(f"\n[{run_name}] Эпоха {epoch} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"Train: Loss = {train_loss:.4f}, Acc = {train_acc:.4f}")
        print(
            f"Val  : Loss = {val_loss:.4f}, Acc = {val_acc:.4f}, "
            f"AUC = {val_auc:.4f}, PR-AUC = {val_prauc:.4f}, Comp = {composite:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_prauc": val_prauc,
            "composite": composite,
            "seed": tm.SEED,
            "baseline": args.encoder,
            "d_model": D_MODEL,
        }
        torch.save(checkpoint, ckpt_last)
        if composite > best_composite:
            best_composite = composite
            torch.save(checkpoint, ckpt_best)
            print(f"Сохранён лучший чекпоинт {ckpt_best} (Comp: {best_composite:.4f})")

    writer.close()
    print(f"\nБазлайн '{args.encoder}' обучен. Best composite = {best_composite:.4f}")


if __name__ == "__main__":
    main()
