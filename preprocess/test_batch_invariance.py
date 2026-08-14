"""Признаки фронтенда не должны зависеть от состава батча.

Регрессия на протечку паддинга: до правки последние 4 валидных остатка
короткой структуры читали координаты паддинга (нули соседа по батчу), и
psi, ca_dist, ram_outliers, local_bending и все 16 PCA-компонент у них
менялись от того, с кем структура попала в батч. На реальной структуре
(3MYC, soft_model) предсказанный scRMSD уезжал на +0.21 Å.
"""

import torch

from preprocess.biophys_frontend import BiophysicalFrontend
from preprocess.test_geometry import _make_helix_coords

TOL = 1e-5  # порядок float32-шума cdist на тензорах разной формы


def _node_feats(frontend, coords: torch.Tensor, length: int, pad_to: int) -> torch.Tensor:
    """Признаки одной структуры, положенной в тензор длины pad_to."""
    padded = torch.zeros(1, pad_to, 3, 3)
    padded[0, :length] = coords[0, :length]
    mask = torch.zeros(1, pad_to, dtype=torch.bool)
    mask[0, :length] = True
    return frontend(padded, mask)["node_feats"][0, :length]


def test_padding_does_not_change_features(device):
    frontend = BiophysicalFrontend(use_no_grad=True).to(device).eval()
    length = 40
    coords = _make_helix_coords(1, length, device)

    reference = _node_feats(frontend, coords, length, length)
    for extra in (1, 5, 50):
        padded = _node_feats(frontend, coords, length, length + extra)
        assert torch.allclose(reference, padded, atol=TOL), (
            f"паддинг +{extra} изменил признаки: "
            f"макс |Δ| = {(reference - padded).abs().max():.2e}"
        )


def test_batch_composition_does_not_change_features(device):
    frontend = BiophysicalFrontend(use_no_grad=True).to(device).eval()
    lengths = [20, 33, 47]
    samples = [_make_helix_coords(1, n, device) for n in lengths]

    n_max = max(lengths)
    coords = torch.zeros(len(lengths), n_max, 3, 3, device=device)
    mask = torch.zeros(len(lengths), n_max, dtype=torch.bool, device=device)
    for i, (n, s) in enumerate(zip(lengths, samples)):
        coords[i, :n] = s[0]
        mask[i, :n] = True
    batched = frontend(coords, mask)["node_feats"]

    for i, (n, s) in enumerate(zip(lengths, samples)):
        alone = _node_feats(frontend, s, n, n)
        assert torch.allclose(alone, batched[i, :n], atol=TOL), (
            f"структура L={n} в батче считается иначе, чем в одиночку: "
            f"макс |Δ| = {(alone - batched[i, :n]).abs().max():.2e}"
        )
