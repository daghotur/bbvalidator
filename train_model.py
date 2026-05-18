import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Импорты наших модулей (настройте пути согласно вашей структуре)
from dataset.dataloader import get_dataloaders
from preprocess.biophys_frontend import BiophysicalFrontend
from model.encoder import HybridProteinEncoder
from model.heads_loss import (
    MultiHeadAttentionPooling,
    ProteinMultiTaskHeads,
    DynamicMultiTaskLoss,
    ProteinScoreModel
)


def train_one_epoch(model, criterion, loader, optimizer, device, epoch, writer, global_step, scaler):
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
        # Очистка градиентов
        optimizer.zero_grad(set_to_none=True)
        # Forward pass в смешанной точности (bfloat16)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = model(coords, mask)
            loss, metrics = criterion(preds, batch_device)

        # Backward pass через GradScaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Шаг оптимизатора
        scaler.step(optimizer)
        scaler.update()

        # Статистика (все тензоры извлекаем как item())
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
    task_names = ["Fold", "RMSD", "Steric", "HBond", "FailMode"]
    for name, w in zip(task_names, weights): writer.add_scalar(f"Loss_Weights/{name}", w, epoch)

    return total_loss / len(loader), epoch_acc, global_step


@torch.no_grad()
def validate(model, criterion, loader, device, epoch, writer):
    model.eval()
    criterion.eval()

    total_loss = 0.0
    correct_fold = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for batch in pbar:
        batch_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        coords = batch_device["coords"]
        mask = batch_device["mask"]

        preds = model(coords, mask)
        loss, metrics = criterion(preds, batch_device)

        total_loss += loss.item()
        fold_preds = (torch.sigmoid(preds["fold_logit"]) > 0.5).float()
        correct_fold += (fold_preds == batch_device["label"].squeeze(-1)).sum().item()
        total_samples += batch_device["label"].size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{correct_fold / total_samples:.3f}",
        })

    avg_loss = total_loss / len(loader)
    epoch_acc = correct_fold / total_samples

    writer.add_scalar("Val/Loss_Total", avg_loss, epoch)
    writer.add_scalar("Val/Accuracy", epoch_acc, epoch)

    return avg_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Инициализация на устройстве: {device}")
    d_model = 192
    frontend = BiophysicalFrontend(use_no_grad=True)
    encoder = HybridProteinEncoder(
        node_in_dim=31,
        pair_in_dim=20,
        d_model=d_model,
        pair_dim=64,
        num_graph_layers=2,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.15
    )
    pooler = MultiHeadAttentionPooling(d_model=d_model, num_heads=4)
    heads = ProteinMultiTaskHeads(d_model=d_model, dropout=0.15)

    model = ProteinScoreModel(frontend, encoder, pooler, heads).to(device)

    # 2. Инициализация критерия (динамический лосс)
    criterion = DynamicMultiTaskLoss(num_tasks=5).to(device)

    # 3. Оптимизатор объединяет параметры сети И параметры лосса
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 3e-4, 'weight_decay': 1e-4},
        {'params': criterion.parameters(), 'lr': 1e-3}
    ])

    # Планировщик learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    # 4. Данные
    manifest_path = "dataset/manifest_v1_split.csv"
    print(f"Загрузка данных из {manifest_path}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        manifest_path,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )

    # 5. Логирование и чекпоинты
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/ProteinScoreModel")

    num_epochs = 15
    best_val_acc = 0.0
    global_step = 0

    scaler = torch.amp.GradScaler('cuda')

    print(f"Старт обучения ({num_epochs} эпох)...")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, global_step = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, writer, global_step, scaler
        )

        val_loss, val_acc = validate(
            model, criterion, val_loader, device, epoch, writer
        )

        scheduler.step()

        print(f"\nЭпоха {epoch} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"Train: Loss = {train_loss:.4f}, Acc = {train_acc:.4f}")
        print(f"Val  : Loss = {val_loss:.4f}, Acc = {val_acc:.4f}")

        # Сохранение лучших весов
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"checkpoints/best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"Сохранен новый лучший чекпоинт: {save_path} (Acc: {best_val_acc:.4f})")

    writer.close()


if __name__ == "__main__":
    main()
