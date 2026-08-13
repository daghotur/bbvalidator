"""
train_model.py
--------------
Обучение ProteinScoreModel (HybridProteinEncoder + мультизадачные головы).

Воспроизводимость: фиксированный SEED (torch/numpy/random, сид сэмплера и
worker'ов). Побитовая идентичность на GPU не гарантируется (cudnn.benchmark),
но конфигурация полностью документируется в чекпоинте.

Селекция модели: составная метрика (Val ROC-AUC + Val PR-AUC) / 2 —
порог-независимая и устойчивая к дисбалансу классов 1:2.
"""

import contextlib
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataloader import get_dataloaders
from model.encoder import HybridProteinEncoder
from model.heads_loss import (
    DynamicMultiTaskLoss,
    MultiHeadAttentionPooling,
    ProteinMultiTaskHeads,
    ProteinScoreModel,
)
from model.metrics import average_precision_score, roc_auc_score
from preprocess.biophys_frontend import BiophysicalFrontend
from preprocess.pair_features import PairFeatureBuilder
from preprocess.fit_pca import load_pca_into_frontend

SEED = 42
NUM_TASKS = 4  # fold, rmsd, steric, failure_mode (без hbond)
TASK_NAMES = ["Fold", "RMSD", "Steric", "FailMode"]
NUM_FAILURE_CLASSES = 6  # класс 5 зарезервирован под OOD-негативы

MANIFEST_PATH = "dataset/manifest_v1_split.csv"
PCA_PATH = "dataset/pca_components.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast_ctx(device: torch.device):
    """bfloat16-автокаст на CUDA, иначе — no-op."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_failure_class_weights(manifest_path: str) -> torch.Tensor:
    """Обратная частота классов failure_mode по train-сплиту, нормировка на сумму."""
    df = pd.read_csv(manifest_path)
    train = df[df["split"] == "train"]
    counts = (
        train["failure_mode_label"]
        .value_counts()
        .reindex(range(NUM_FAILURE_CLASSES), fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    weights = np.zeros(NUM_FAILURE_CLASSES, dtype=np.float64)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    weights /= weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, criterion, loader, optimizer, device, epoch, writer, global_step):
    model.train()
    criterion.train()

    total_loss = 0.0
    correct_fold = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        batch_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        coords = batch_device["coords"]
        mask = batch_device["mask"]

        optimizer.zero_grad(set_to_none=True)
        # bf16 имеет тот же диапазон экспоненты, что и fp32, поэтому GradScaler не нужен.
        with _autocast_ctx(device):
            preds = model(coords, mask)
            loss, metrics = criterion(preds, batch_device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        fold_preds = (torch.sigmoid(preds["fold_logit"].float()) > 0.5).float()
        correct_fold += (fold_preds == batch_device["label"].squeeze(-1)).sum().item()
        total_samples += batch_device["label"].size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{correct_fold / total_samples:.3f}",
        })

        writer.add_scalar("Train/Loss_Total", loss.item(), global_step)
        for k, v in metrics.items():
            if k != "weights":
                writer.add_scalar(f"Train_Details/{k}", v, global_step)

        global_step += 1

    epoch_acc = correct_fold / total_samples

    weights = metrics["weights"]
    for name, w in zip(TASK_NAMES, weights):
        writer.add_scalar(f"Loss_Weights/{name}", w, epoch)

    return total_loss / len(loader), epoch_acc, global_step


@torch.no_grad()
def validate(model, criterion, loader, device, epoch, writer):
    model.eval()
    criterion.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for batch in pbar:
        batch_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        coords = batch_device["coords"]
        mask = batch_device["mask"]

        # Val в том же dtype, что и train (bf16 на CUDA) — числа сопоставимы.
        with _autocast_ctx(device):
            preds = model(coords, mask)
            loss, _ = criterion(preds, batch_device)

        total_loss += loss.item()
        probs = torch.sigmoid(preds["fold_logit"].float()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch_device["label"].squeeze(-1).cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    probs_all = np.concatenate(all_probs)
    labels_all = np.concatenate(all_labels)

    avg_loss = total_loss / len(loader)
    epoch_acc = float(((probs_all > 0.5).astype(np.int64) == labels_all).mean())
    val_auc = roc_auc_score(labels_all, probs_all)
    val_prauc = average_precision_score(labels_all, probs_all)
    composite = 0.5 * (val_auc + val_prauc)

    writer.add_scalar("Val/Loss_Total", avg_loss, epoch)
    writer.add_scalar("Val/Accuracy", epoch_acc, epoch)
    writer.add_scalar("Val/ROC_AUC", val_auc, epoch)
    writer.add_scalar("Val/PR_AUC", val_prauc, epoch)
    writer.add_scalar("Val/Composite_AUC", composite, epoch)

    return avg_loss, epoch_acc, val_auc, val_prauc, composite


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Инициализация на устройстве: {device} (seed={SEED})")

    d_model = 192
    config = {
        "seed": SEED,
        "d_model": d_model,
        "node_in_dim": 31,
        # синхронизировано с PairFeatureBuilder.feature_dim (rbf+seq+ориентация)
        "pair_in_dim": PairFeatureBuilder().feature_dim,
        "pair_dim": 64,
        "num_graph_layers": 2,
        "num_transformer_layers": 4,
        "num_encoder_heads": 8,
        "num_pool_heads": 4,
        "dropout": 0.15,
        "num_tasks": NUM_TASKS,
        "num_failure_classes": NUM_FAILURE_CLASSES,
        "batch_size": 64,
        "lr_model": 6e-4,
        "lr_loss": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": 15,
        "pca_path": PCA_PATH,
        "manifest_path": MANIFEST_PATH,
    }

    # 1. Модель: фронтенд с честной PCA (fit + заморозка)
    frontend = BiophysicalFrontend(use_no_grad=True)
    load_pca_into_frontend(frontend, PCA_PATH)

    encoder = HybridProteinEncoder(
        node_in_dim=config["node_in_dim"],
        pair_in_dim=config["pair_in_dim"],
        d_model=d_model,
        pair_dim=config["pair_dim"],
        num_graph_layers=config["num_graph_layers"],
        num_transformer_layers=config["num_transformer_layers"],
        num_heads=config["num_encoder_heads"],
        dropout=config["dropout"],
    )
    pooler = MultiHeadAttentionPooling(d_model=d_model, num_heads=config["num_pool_heads"])
    heads = ProteinMultiTaskHeads(d_model=d_model, dropout=config["dropout"])

    model = ProteinScoreModel(frontend, encoder, pooler, heads).to(device)

    # 2. Критерий: 4 задачи, веса классов failure_mode по обратной частоте
    class_weights = load_failure_class_weights(MANIFEST_PATH)
    print(f"Class weights failure_mode: {class_weights.numpy().round(4)}")
    criterion = DynamicMultiTaskLoss(
        num_tasks=NUM_TASKS, failure_class_weights=class_weights
    ).to(device)

    # 3. Оптимизатор: только обучаемые параметры сети + параметры лосса
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": trainable, "lr": config["lr_model"], "weight_decay": config["weight_decay"]},
        {"params": criterion.parameters(), "lr": config["lr_loss"]},
    ])

    num_epochs = config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    # 4. Данные (бакетизация по длине, сид сэмплера = SEED)
    print(f"Загрузка данных из {MANIFEST_PATH}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        MANIFEST_PATH,
        batch_size=config["batch_size"],
        num_workers=8,
        pin_memory=True,
        seed=SEED,
    )

    # 5. Логирование и чекпоинты
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/ProteinScoreModel")

    best_composite = 0.0
    global_step = 0

    print(f"Старт обучения ({num_epochs} эпох)...")

    for epoch in range(1, num_epochs + 1):
        # Новый рисунок батчей каждую эпоху (бакетный сэмплер)
        train_loader.batch_sampler.set_epoch(epoch)

        train_loss, train_acc, global_step = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, writer, global_step
        )

        val_loss, val_acc, val_auc, val_prauc, composite = validate(
            model, criterion, val_loader, device, epoch, writer
        )

        scheduler.step()

        print(f"\nЭпоха {epoch} | LR: {scheduler.get_last_lr()[0]:.2e}")
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
            "seed": SEED,
            "config": config,
        }

        # Last — всегда, для возобновления
        torch.save(checkpoint, "checkpoints/last_model.pth")

        # Best — по составной AUC-метрике
        if composite > best_composite:
            best_composite = composite
            torch.save(checkpoint, "checkpoints/best_model.pth")
            print(f"Сохранён новый лучший чекпоинт (Comp: {best_composite:.4f})")

    writer.close()
    print(f"\nОбучение завершено. Best composite = {best_composite:.4f}")


if __name__ == "__main__":
    main()
