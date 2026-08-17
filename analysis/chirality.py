"""
analysis/chirality.py
---------------------
Проверка хиральности скорера: зеркальный остов физически несворачиваем (белки
состоят из L-аминокислот), поэтому модель обязана оценивать отражённую структуру
хуже оригинала.

Матрица расстояний Cα к отражению инвариантна, так что различить зеркало можно
только по ориентационным признакам рёбер (набор trRosetta, docs/05 раздел 5.8).
Скрипт меряет две величины на одной и той же выборке остовов:

* `spearman_orig_mirror` — сколько порядка переживает отражение. В идеале около
  нуля: ранжирование физически несуществующих структур не должно повторять
  ранжирование настоящих.
* `frac_mirror_worse` — доля структур, у которых зеркало получило худший скор.
  В идеале 1.0.

Зачем отдельный скрипт. Числа §5.8 (Spearman 0.037, доля 100%) получены разовым
прогоном, который не сохранился, и на действующих чекпоинтах не воспроизводятся.
Проверка должна быть исполняемой, иначе такие расхождения всплывают случайно.

Замечание о загрузке. Чекпоинты до расширения парного канала 20 → 29 грузятся
в текущую архитектуру со случайной инициализацией девяти ориентационных входов
(`inference.build_model` печатает предупреждение). Их числа к делу не относятся:
меряется не обученная модель. Такие случаи помечаются `loaded_cleanly = false`.

Запуск:  python -m analysis.chirality
"""

import argparse
import contextlib
import glob
import io
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from inference import build_model, center_coords, parse_pdb_to_backbone

SCAFFOLDS = "data/ood/rfdiffusion/scaffolds/**/*.pdb"
DEFAULT_CKPTS = [
    ("checkpoints/soft_model.pth", False),
    ("checkpoints/soft_model_mt.pth", False),
    ("checkpoints/joint_model.pth", True),
]


def load_coords(n: int, seed: int) -> list[torch.Tensor]:
    paths = sorted(glob.glob(SCAFFOLDS, recursive=True))
    if not paths:
        raise SystemExit(f"нет структур по маске {SCAFFOLDS}")
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(paths)[:n]
    return [torch.from_numpy(center_coords(parse_pdb_to_backbone(p))).float()
            for p in chosen]


def measure(ckpt: str, per_residue: bool, coords, device) -> dict:
    # build_model печатает диагностику загрузки; перехватываем её, чтобы отличить
    # чистую загрузку от расширения входа под старый чекпоинт.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = build_model(ckpt, device, per_residue=per_residue,
                            pair_init="scaled")
    model.eval()
    clean = "вход расширен" not in buf.getvalue()

    orig, mirror = [], []
    for c in coords:
        x = c.unsqueeze(0).to(device)
        mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
        mir = x.clone()
        mir[..., 2] *= -1.0                      # отражение по одной оси
        with torch.no_grad():
            orig.append(model(x, mask)["fold_logit"].float().item())
            mirror.append(model(mir, mask)["fold_logit"].float().item())

    o, m = np.array(orig), np.array(mirror)
    # Выход — log1p(scRMSD): больше значит хуже.
    return {
        "checkpoint": ckpt,
        "loaded_cleanly": clean,
        "n_structures": len(coords),
        "spearman_orig_mirror": float(spearmanr(o, m).statistic),
        "frac_mirror_worse": float((m > o).mean()),
        "median_penalty_A": float(np.median(np.expm1(m) - np.expm1(o))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=600, help="структур в выборке")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="пути к чекпоинтам; по умолчанию — рабочий набор")
    ap.add_argument("-o", default="results/analysis_chirality.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords = load_coords(a.n, a.seed)
    print(f"структур: {len(coords)}, устройство: {device}\n")

    todo = ([(c, "joint" in c) for c in a.ckpt] if a.ckpt else DEFAULT_CKPTS)
    rows = []
    for ckpt, per_res in todo:
        if not os.path.exists(ckpt):
            print(f"нет файла: {ckpt}")
            continue
        r = measure(ckpt, per_res, coords, device)
        rows.append(r)
        flag = "" if r["loaded_cleanly"] else "  [загружен с расширением входа]"
        print(f"{os.path.basename(ckpt):24s} Spearman {r['spearman_orig_mirror']:6.3f}   "
              f"зеркал хуже {r['frac_mirror_worse']:5.1%}   "
              f"штраф {r['median_penalty_A']:+.2f} A{flag}")

    os.makedirs(os.path.dirname(a.o) or ".", exist_ok=True)
    with open(a.o, "w", encoding="utf-8") as f:
        json.dump({"n_structures": a.n, "seed": a.seed, "checkpoints": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n-> {a.o}")


if __name__ == "__main__":
    main()
