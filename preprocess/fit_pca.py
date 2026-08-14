"""
preprocess/fit_pca.py
---------------------
Одноразовый скрипт: честная PCA по попарным расстояниям 9-остаточных
фрагментов из нативных структур (positive_proteins.h5).

Один проход по датасету без хранения всех фрагментов в памяти:
накапливаются сумма и матрица вторых моментов (36×36), из которых
восстанавливается точная ковариация. Топ-pca_components собственных
векторов записываются в файл вместе со средним фрагмента.

Базис снимается ТОЛЬКО с train-сплита: он замораживается и становится частью
признаков, поэтому фит по всем нативам подмешивал бы в них геометрию val и
test (трансдуктивная утечка, пусть и без меток).

Запуск (после пересборки данных):
    python -m preprocess.fit_pca \
        --h5 dataset/positive_proteins.h5 \
        --manifest dataset/manifest_v1_split.csv --split train \
        --out dataset/pca_components.pth

Загрузка в модель (training/hybrid.py, inference.py):
    from preprocess.fit_pca import load_pca_into_frontend
    load_pca_into_frontend(frontend, "dataset/pca_components.pth")
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .designability_features import DesignabilityProxies

FRAGMENT_SIZE = 9
FRAG_PAIRS = (FRAGMENT_SIZE * (FRAGMENT_SIZE - 1)) // 2  # 36
PAD = FRAGMENT_SIZE // 2  # 4 позиции с каждого края — replicate-паддинг, не настоящие фрагменты


def split_keys(manifest_path: str | None, split: str) -> set[str] | None:
    """Ключи h5, попавшие в указанный сплит. None — брать все (манифеста нет)."""
    if manifest_path is None:
        return None
    df = pd.read_csv(manifest_path, usecols=["h5_group_key", "split", "label"])
    return set(df[(df["split"] == split) & (df["label"] == 1)]["h5_group_key"].astype(str))


@torch.no_grad()
def _iter_fragment_distances(
    h5_path: str,
    proxy: DesignabilityProxies,
    device: torch.device,
    keys: set[str] | None = None,
):
    """Генератор тензоров [M, 36] — расстояния настоящих (не паддинговых) фрагментов."""
    with h5py.File(h5_path, "r") as h5f:
        for key in h5f.keys():
            if keys is not None and key not in keys:
                continue
            grp = h5f[key]
            if "coords" not in grp:
                continue
            coords = torch.from_numpy(grp["coords"][:].astype(np.float32))
            L = coords.shape[0]
            if L <= FRAGMENT_SIZE:
                continue
            ca = coords[:, 1, :].unsqueeze(0).to(device)  # [1, L, 3]
            frag = proxy._get_local_pairwise_distances(ca)[0]  # [L, 36]
            yield frag[PAD : L - PAD].cpu()


def fit_pca(
    h5_path: str,
    pca_components: int = 16,
    device: torch.device | None = None,
    keys: set[str] | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proxy = DesignabilityProxies(
        pca_components=pca_components, fragment_size=FRAGMENT_SIZE
    ).to(device)

    sum_x = np.zeros(FRAG_PAIRS, dtype=np.float64)
    second_moment = np.zeros((FRAG_PAIRS, FRAG_PAIRS), dtype=np.float64)
    n_fragments = 0
    n_chains = 0

    for frag in tqdm(_iter_fragment_distances(h5_path, proxy, device, keys), desc="fit PCA"):
        x = frag.numpy().astype(np.float64)
        n_fragments += x.shape[0]
        n_chains += 1
        sum_x += x.sum(axis=0)
        second_moment += x.T @ x

    if n_fragments == 0:
        raise ValueError(f"В {h5_path} не нашлось ни одного фрагмента длины >= {FRAGMENT_SIZE}")

    mean = sum_x / n_fragments
    cov = second_moment / n_fragments - np.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)  # защита от накопленной асимметрии

    eigvals, eigvecs = np.linalg.eigh(cov)  # по возрастанию
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    components = torch.from_numpy(eigvecs[:, :pca_components].T.copy()).float()  # [C, 36]
    explained = float(eigvals[:pca_components].sum() / max(eigvals.sum(), 1e-12))

    return {
        "components": components,
        "frag_mean": torch.from_numpy(mean).float(),
        "n_fragments": int(n_fragments),
        "explained_variance": explained,
        "eigenvalues": torch.from_numpy(eigvals).float(),
        "source_h5": os.path.abspath(h5_path),
        "fragment_size": FRAGMENT_SIZE,
        "n_chains": int(n_chains),
    }


def load_pca_into_frontend(frontend, pca_path: str) -> None:
    """Загружает фит PCA во фронтенд и замораживает проекцию."""
    if not os.path.exists(pca_path):
        raise FileNotFoundError(
            f"PCA-веса не найдены: {os.path.abspath(pca_path)}. "
            "Сначала запустите preprocess/fit_pca.py после пересборки данных."
        )
    state = torch.load(pca_path, map_location="cpu", weights_only=True)

    designability = frontend.designability
    components = state["components"]
    if components.shape != designability.pca_proj.weight.shape:
        raise ValueError(
            f"Форма PCA-весов {tuple(components.shape)} не совпадает с "
            f"ожидаемой {tuple(designability.pca_proj.weight.shape)}"
        )

    designability.pca_proj.weight.data.copy_(components)
    designability.frag_mean.data.copy_(state["frag_mean"])
    designability.freeze_pca()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit PCA фрагментов по нативным структурам")
    parser.add_argument("--h5", default="dataset/positive_proteins.h5")
    parser.add_argument("--out", default="dataset/pca_components.pth")
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--manifest", default="dataset/manifest_v1_split.csv",
                        help="манифест со сплитами; '' — фит по всем цепям файла")
    parser.add_argument("--split", default="train",
                        help="сплит, по которому снимается базис")
    args = parser.parse_args()

    keys = split_keys(args.manifest or None, args.split)
    if keys is None:
        print("ВНИМАНИЕ: фит по всем цепям — базис увидит val и test.")
    state = fit_pca(args.h5, pca_components=args.components, keys=keys)
    state["split"] = args.split if keys is not None else "all"
    print(
        f"Цепей: {state['n_chains']:,} (сплит {state['split']}) | "
        f"фрагментов: {state['n_fragments']:,} | "
        f"объяснённая дисперсия (топ-{args.components}): {state['explained_variance']:.4f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(state, args.out)
    print(f"PCA-веса сохранены в {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
