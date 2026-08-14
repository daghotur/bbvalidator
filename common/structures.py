"""
common/structures.py
--------------------
Чтение PDB-структур с диска и сборка их в батчи.

Батчевый цикл «отсортировать по длине → сложить в нулевой тензор → маска»
жил в одиннадцати копиях по analysis/, training/, evaluation/ и фильтру.
Здесь он один: паддинг — место, где легко получить зависимость признаков от
состава батча, и такое место должно быть ровно одно.
"""

import glob
import os
from typing import Iterator, Sequence

import numpy as np
import torch

from inference import center_coords, parse_pdb_to_backbone


def collect_pdbs(input_path: str, pattern: str = "**/*.pdb") -> list[str]:
    """Один файл или все PDB внутри директории (без мусора macOS-архивов)."""
    if os.path.isfile(input_path):
        return [input_path]
    files = sorted(glob.glob(os.path.join(input_path, pattern), recursive=True))
    return [
        f
        for f in files
        if "__MACOSX" not in f and not os.path.basename(f).startswith("._")
    ]


def parse_pdb_files(root_dir: str) -> tuple[list[dict], int]:
    """Структуры генератора: [{sample, motif, coords}], координаты центрированы.

    Возвращает (records, число непрочитанных файлов). Раскладка MotifBench:
    <root>/<motif>/<sample>.pdb.
    """
    records = []
    skipped = 0
    for path in collect_pdbs(root_dir):
        try:
            coords = parse_pdb_to_backbone(path)
        except Exception:
            skipped += 1
            continue
        parts = path.split(os.sep)
        records.append(
            {
                "sample": parts[-1],
                "motif": parts[-2],
                "coords": center_coords(coords).astype(np.float32),
            }
        )
    return records, skipped


def motifs_in(root_dir: str) -> set[str]:
    """Имена мотивов в каталоге генератора — по раскладке файлов, без парсинга."""
    return {path.split(os.sep)[-2] for path in collect_pdbs(root_dir)}


def batch_order(
    records: Sequence[dict],
    order: str = "length",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Порядок обхода: по длине (меньше паддинга), случайный или как есть."""
    n = len(records)
    if order == "length":
        return np.array(sorted(range(n), key=lambda i: len(records[i]["coords"])))
    if order == "shuffle":
        return rng.permutation(n) if rng is not None else np.random.permutation(n)
    if order == "sequential":
        return np.arange(n)
    raise ValueError(f"order: ожидается length|shuffle|sequential, получено {order!r}")


def iter_padded_batches(
    records: Sequence[dict],
    batch_size: int,
    device: torch.device,
    order: str = "length",
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, torch.Tensor, torch.Tensor]]:
    """Батчи (индексы, coords [B, Lmax, 3, 3], mask [B, Lmax]) на device.

    Индексы отдаются наружу, чтобы вызывающий сам разложил свои таргеты и
    результаты обратно в исходном порядке записей.
    """
    idx_order = batch_order(records, order, rng)
    for start in range(0, len(idx_order), batch_size):
        idxs = idx_order[start : start + batch_size]
        lengths = [len(records[i]["coords"]) for i in idxs]
        coords = np.zeros((len(idxs), max(lengths), 3, 3), dtype=np.float32)
        mask = np.zeros((len(idxs), max(lengths)), dtype=np.bool_)
        for b, (i, length) in enumerate(zip(idxs, lengths)):
            coords[b, :length] = records[i]["coords"]
            mask[b, :length] = True
        yield (
            idxs,
            torch.from_numpy(coords).to(device),
            torch.from_numpy(mask).to(device),
        )
