"""
preprocess/fit_pca.py
---------------------
Одноразовый скрипт: честная PCA по попарным расстояниям 9-остаточных
фрагментов из нативных структур (positive_proteins.h5).

Один проход по датасету без хранения всех фрагментов в памяти:
накапливаются сумма и матрица вторых моментов (36×36), из которых
восстанавливается точная ковариация. Топ-pca_components собственных
векторов записываются в файл вместе со средним фрагмента.

Запуск (после пересборки данных):
    python preprocess/fit_pca.py \
        --h5 dataset/positive_proteins.h5 \
        --out dataset/pca_components.pth

Загрузка в модель (train_model.py, inference.py):
    from preprocess.fit_pca import load_pca_into_frontend
    load_pca_into_frontend(frontend, "dataset/pca_components.pth")
"""

import argparse
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .designability_features import DesignabilityProxies

FRAGMENT_SIZE = 9
FRAG_PAIRS = (FRAGMENT_SIZE * (FRAGMENT_SIZE - 1)) // 2  # 36
PAD = FRAGMENT_SIZE // 2  # 4 позиции с каждого края — replicate-паддинг, не настоящие фрагменты


@torch.no_grad()
def _iter_fragment_distances(
    h5_path: str, proxy: DesignabilityProxies, device: torch.device
):
    """Генератор тензоров [M, 36] — расстояния настоящих (не паддинговых) фрагментов."""
    with h5py.File(h5_path, "r") as h5f:
        for key in h5f.keys():
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
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proxy = DesignabilityProxies(
        pca_components=pca_components, fragment_size=FRAGMENT_SIZE
    ).to(device)

    sum_x = np.zeros(FRAG_PAIRS, dtype=np.float64)
    second_moment = np.zeros((FRAG_PAIRS, FRAG_PAIRS), dtype=np.float64)
    n_fragments = 0

    for frag in tqdm(_iter_fragment_distances(h5_path, proxy, device), desc="fit PCA"):
        x = frag.numpy().astype(np.float64)
        n_fragments += x.shape[0]
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
    args = parser.parse_args()

    state = fit_pca(args.h5, pca_components=args.components)
    print(
        f"Фрагментов: {state['n_fragments']:,} | "
        f"объяснённая дисперсия (топ-{args.components}): {state['explained_variance']:.4f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(state, args.out)
    print(f"PCA-веса сохранены в {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
