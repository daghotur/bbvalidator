"""
eval_model.py
-------------
Оценка обученной модели на указанном сплите (по умолчанию test):
  • fold-задача: Accuracy, ROC-AUC, PR-AUC, ECE (калибровка)
  • per-strategy: точность/полнота бинарного fold-предсказания по каждой
    стратегии декоев и по нативам
  • failure_mode: confusion matrix и accuracy
  • auxiliary: MSE голов rmsd (log1p) и steric

Результаты пишутся в JSON и печатаются markdown-таблицей.

Пример:
    python eval_model.py -c checkpoints/best_model.pth --split test \
        -o eval_results.json
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from baselines.train_baseline import build_baseline
from dataset.dataloader import make_loader
from inference import _autocast_ctx, build_model, remap_legacy_foldability_keys
from model.metrics import average_precision_score, expected_calibration_error, roc_auc_score
from preprocess.fit_pca import load_pca_into_frontend

NUM_FAILURE_CLASSES = 6
THRESHOLD = 0.5


def build_eval_model(ckpt_path: str, device: torch.device, pca_path: str):
    """Строит модель под архитектуру чекпоинта (гибрид / MLP / GPS).

    Архитектура определяется по ключам state_dict: encoder.net.* — MLP,
    encoder.gps_layers.* — GPS, encoder.graph_layers.* — гибрид.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = remap_legacy_foldability_keys(state_dict)

    if any(k.startswith("encoder.net.") for k in state_dict):
        arch = "mlp"
    elif any(k.startswith("encoder.gps_layers.") for k in state_dict):
        arch = "gps"
    else:
        arch = "hybrid"

    print(f"Архитектура чекпоинта: {arch}")

    if arch == "hybrid":
        return build_model(ckpt_path, device, pca_path=pca_path), arch

    model = build_baseline(arch, device)
    if os.path.exists(pca_path):
        load_pca_into_frontend(model.frontend, pca_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, arch


@torch.no_grad()
def run_evaluation(model, loader, device) -> dict:
    model.eval()

    probs_all = []
    labels_all = []
    strategies_all = []
    fail_preds_all = []
    fail_labels_all = []
    rmsd_preds_all = []
    rmsd_targets_all = []
    steric_preds_all = []
    steric_targets_all = []

    for batch in tqdm(loader, desc="Eval"):
        batch_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with _autocast_ctx(device):
            preds = model(batch_device["coords"], batch_device["mask"])

        probs_all.append(torch.sigmoid(preds["fold_logit"].float()).cpu().numpy())
        labels_all.append(batch_device["label"].squeeze(-1).cpu().numpy())
        strategies_all.extend(batch_device["strategy"])
        fail_preds_all.append(preds["failure_mode"].argmax(dim=-1).cpu().numpy())
        fail_labels_all.append(batch_device["failure_mode_label"].cpu().numpy())
        rmsd_preds_all.append(preds["rmsd"].float().cpu().numpy())
        rmsd_targets_all.append(batch_device["rmsd_target"].cpu().numpy())
        steric_preds_all.append(preds["steric"].float().cpu().numpy())
        steric_targets_all.append(batch_device["steric_target"].cpu().numpy())

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all).astype(np.int64)
    preds_bin = (probs > THRESHOLD).astype(np.int64)
    strategies = np.array(strategies_all)
    fail_preds = np.concatenate(fail_preds_all)
    fail_labels = np.concatenate(fail_labels_all)
    rmsd_preds = np.concatenate(rmsd_preds_all)
    rmsd_targets = np.concatenate(rmsd_targets_all)
    steric_preds = np.concatenate(steric_preds_all)
    steric_targets = np.concatenate(steric_targets_all)

    # --- fold-батарея ---
    tp = int(((preds_bin == 1) & (labels == 1)).sum())
    fp = int(((preds_bin == 1) & (labels == 0)).sum())
    fn = int(((preds_bin == 0) & (labels == 1)).sum())
    tn = int(((preds_bin == 0) & (labels == 0)).sum())

    fold_metrics = {
        "n_samples": int(len(labels)),
        "accuracy": float((preds_bin == labels).mean()),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "ece": expected_calibration_error(labels, probs),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    # --- разбивка по стратегиям ---
    per_strategy = {}
    for strat in sorted(set(strategies)):
        m = strategies == strat
        s_labels = labels[m]
        s_preds = preds_bin[m]
        n_pos = int((s_labels == 1).sum())
        correct = int((s_preds == s_labels).sum())
        if n_pos == len(s_labels):
            # нативы: интересна полнота (доля распознанных как foldable)
            recall = float((s_preds == 1).mean())
            per_strategy[str(strat)] = {
                "n": int(m.sum()),
                "recall": recall,
                "accuracy": correct / int(m.sum()),
            }
        else:
            # декoi: доля корректно отвергнутых (specificity) + AUC
            # «нативы против данного семейства декоев» — мера трудности семейства
            specificity = float((s_preds == 0).mean())
            pair_mask = m | (labels == 1)
            per_strategy[str(strat)] = {
                "n": int(m.sum()),
                "specificity": specificity,
                "roc_auc": roc_auc_score(labels[pair_mask], probs[pair_mask]),
                "accuracy": correct / int(m.sum()),
            }

    # --- failure_mode ---
    confusion = np.zeros((NUM_FAILURE_CLASSES, NUM_FAILURE_CLASSES), dtype=np.int64)
    for t, p in zip(fail_labels, fail_preds):
        confusion[t, p] += 1
    failure_metrics = {
        "accuracy": float((fail_preds == fail_labels).mean()),
        "confusion": confusion.tolist(),
    }

    # --- auxiliary головы ---
    aux_metrics = {
        "rmsd_mse_log1p": float(np.mean((rmsd_preds - np.log1p(rmsd_targets)) ** 2)),
        "steric_mse": float(np.mean((steric_preds - steric_targets) ** 2)),
    }

    return {
        "fold": fold_metrics,
        "per_strategy": per_strategy,
        "failure_mode": failure_metrics,
        "aux": aux_metrics,
    }


def print_markdown(results: dict) -> None:
    f = results["fold"]
    print("\n=== FOLD (бинарная задача) ===")
    print("| Метрика | Значение |")
    print("|---|---|")
    print(f"| Accuracy | {f['accuracy']:.4f} |")
    print(f"| ROC-AUC | {f['roc_auc']:.4f} |")
    print(f"| PR-AUC | {f['pr_auc']:.4f} |")
    print(f"| ECE | {f['ece']:.4f} |")
    print(f"| Precision | {f['precision']:.4f} |")
    print(f"| Recall | {f['recall']:.4f} |")
    print(f"| TP/FP/FN/TN | {f['tp']}/{f['fp']}/{f['fn']}/{f['tn']} |")

    print("\n=== PER-STRATEGY ===")
    print("| Стратегия | n | Accuracy | Recall / Specificity | AUC |")
    print("|---|---|---|---|---|")
    for strat, m in results["per_strategy"].items():
        if "recall" in m:
            print(f"| {strat} | {m['n']} | {m['accuracy']:.4f} | recall {m['recall']:.4f} | — |")
        else:
            print(f"| {strat} | {m['n']} | {m['accuracy']:.4f} | spec {m['specificity']:.4f} | {m['roc_auc']:.4f} |")

    fm = results["failure_mode"]
    print(f"\n=== FAILURE MODE: accuracy = {fm['accuracy']:.4f} ===")
    header = "true\\pred | " + " ".join(str(i) for i in range(NUM_FAILURE_CLASSES))
    print(header)
    for i, row in enumerate(fm["confusion"]):
        print(f"{i} | " + " ".join(str(v) for v in row))

    a = results["aux"]
    print(f"\n=== AUX: rmsd_mse_log1p = {a['rmsd_mse_log1p']:.4f}, steric_mse = {a['steric_mse']:.4f} ===")


def main():
    parser = argparse.ArgumentParser(description="Оценка ProteinScoreModel на сплите")
    parser.add_argument("-c", "--ckpt", default="checkpoints/best_model.pth")
    parser.add_argument("--manifest", default="dataset/manifest_v1_split.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pca", default="dataset/pca_components.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("-o", "--output", default="eval_results.json")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Оценка на устройстве: {device} | split={args.split}")

    model, arch = build_eval_model(args.ckpt, device, pca_path=args.pca)

    loader = make_loader(
        args.manifest,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    results = run_evaluation(model, loader, device)
    results["architecture"] = arch
    results["split"] = args.split
    results["checkpoint"] = os.path.abspath(args.ckpt)
    results["manifest"] = os.path.abspath(args.manifest)

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    print_markdown(results)
    print(f"\nРезультаты сохранены в {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
