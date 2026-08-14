"""
common/scoring.py
-----------------
Скоринг структур обученной моделью в единицах «меньше = дизайнируемее».

Одна формула на все анализы: сравнивать lift и Spearman разных режимов
супервизии можно только тогда, когда скор считается одинаково.
"""

import pandas as pd
import torch

from common.structures import iter_padded_batches
from inference import _autocast_ctx


@torch.no_grad()
def score_designability(
    model,
    records: list[dict],
    device: torch.device,
    readout: str = "fold",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Таблица (sample, motif, pred_scrmsd) в порядке записей.

    readout="fold" — предсказанный scRMSD головой fold, expm1(fold_logit);
    readout="lddt" — агрегат по-остаточной головы, со знаком минус, чтобы
    «меньше = лучше» выполнялось и здесь.
    """
    rows: list[dict | None] = [None] * len(records)

    for idxs, coords, mask in iter_padded_batches(records, batch_size, device):
        with _autocast_ctx(device):
            out = model(coords, mask)

        if readout == "lddt":
            lddt = torch.sigmoid(out["lddt_logit"].float())
            valid = mask.float()
            score = -((lddt * valid).sum(-1) / valid.sum(-1).clamp(min=1))
        elif readout == "fold":
            score = torch.clamp(torch.expm1(out["fold_logit"].float()), min=0.0)
        else:
            raise ValueError(f"readout: ожидается fold|lddt, получено {readout!r}")

        score = score.cpu().numpy()
        for b, i in enumerate(idxs):
            rows[i] = {
                "sample": records[i]["sample"].removesuffix(".pdb"),
                "motif": records[i]["motif"],
                "pred_scrmsd": float(score[b]),
            }

    return pd.DataFrame(rows)


def score_lookup(pred: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(motif, sample) -> предсказанный scRMSD."""
    return dict(zip(zip(pred["motif"], pred["sample"]), pred["pred_scrmsd"]))
