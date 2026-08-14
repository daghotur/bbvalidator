"""
model/metrics.py
----------------
Метрики оценки без внешних ML-зависимостей:
  • roc_auc_score           — площадь под ROC-кривой (rank-based, с поправкой на тай)
  • average_precision_score — PR-AUC (интерполяция по PR-кривой)
  • expected_calibration_error — калибровка вероятностей (равные бины)

Используются в цикле селекции модели (training/hybrid.py) и в evaluation/eval_model.py.
"""

import numpy as np
from scipy.stats import rankdata


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC через U-статистику Манна–Уитни: P(score_pos > score_neg)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(scores, method="average")
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC: сумма приращений полноты, взвешенных точностью (без интерполяции вниз)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    n_pos = int((labels == 1).sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / (tp + fp)
    recall = tp / n_pos

    # Приращение полноты на каждом шаге (первый шаг — относительно 0)
    d_recall = np.diff(recall, prepend=0.0)
    return float((precision * d_recall).sum())


def expected_calibration_error(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> float:
    """ECE: взвешенная средняя |точность − уверенность| по равным бинам вероятности."""
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    if n == 0:
        return float("nan")

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probs > lo) & (probs <= hi) if lo > 0 else (probs <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        avg_conf = probs[in_bin].mean()
        avg_acc = labels[in_bin].mean()
        ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)
