"""Тесты парных признаков: корректность углов и хиральность канала."""

import numpy as np
import torch

from preprocess.biophys_frontend import BiophysicalFrontend
from preprocess.pair_features import PairFeatureBuilder


def _builder() -> PairFeatureBuilder:
    return PairFeatureBuilder(rbf_bins=16, k_neighbors=8)


def test_dihedral_known_value():
    """Двугранный угол на эталонной конфигурации равен 90 градусам."""
    pb = _builder()
    p1 = torch.tensor([[1.0, 0.0, 0.0]])
    p2 = torch.tensor([[0.0, 0.0, 0.0]])
    p3 = torch.tensor([[0.0, 0.0, 1.0]])
    p4 = torch.tensor([[0.0, 1.0, 1.0]])
    ang = pb._dihedral(p1, p2, p3, p4)
    assert np.isclose(abs(float(ang)), np.pi / 2, atol=1e-5)


def test_dihedral_flips_sign_under_reflection():
    """Отражение меняет знак двугранного угла — основа хиральности канала."""
    pb = _builder()
    g = torch.Generator().manual_seed(0)
    pts = [torch.randn(32, 3, generator=g) for _ in range(4)]
    direct = pb._dihedral(*pts)
    flip = torch.tensor([1.0, 1.0, -1.0])
    mirrored = pb._dihedral(*[p * flip for p in pts])
    assert torch.allclose(direct, -mirrored, atol=1e-5)


def test_angle_known_value():
    """Планарный угол: перпендикуляр даёт косинус 0, развёрнутый — минус 1."""
    pb = _builder()
    v = torch.tensor([[0.0, 0.0, 0.0]])
    a = torch.tensor([[1.0, 0.0, 0.0]])
    assert np.isclose(float(pb._angle(a, v, torch.tensor([[0.0, 1.0, 0.0]]))), 0.0, atol=1e-6)
    assert np.isclose(float(pb._angle(a, v, torch.tensor([[-1.0, 0.0, 0.0]]))), -1.0, atol=1e-6)


def test_feature_dim_matches_output():
    """Заявленная размерность совпадает с фактической."""
    pb = _builder()
    g = torch.Generator().manual_seed(1)
    coords = torch.randn(2, 24, 3, 3, generator=g) * 5
    mask = torch.ones(2, 24, dtype=torch.bool)
    out = pb(coords[:, :, 1, :], mask,
             cb_coords=coords[:, :, 2, :], n_coords=coords[:, :, 0, :])
    assert out["edge_attr"].shape[1] == pb.feature_dim


def test_distance_block_is_mirror_invariant_orientation_is_not():
    """Ключевое свойство: расстояния и seq-признаки к отражению инвариантны,
    ориентационные — нет. Именно это чинит ахиральность парного канала."""
    fe = BiophysicalFrontend()
    g = torch.Generator().manual_seed(2)
    coords = torch.randn(1, 40, 3, 3, generator=g) * 6
    mask = torch.ones(1, 40, dtype=torch.bool)

    mirrored = coords.clone()
    mirrored[..., 2] *= -1.0

    with torch.no_grad():
        a = fe(coords, mask)["edge_attr"]
        b = fe(mirrored, mask)["edge_attr"]

    n_geo = a.shape[1] - PairFeatureBuilder.ORIENT_DIM
    assert torch.allclose(a[:, :n_geo], b[:, :n_geo], atol=1e-4), \
        "блок расстояний и последовательности обязан быть инвариантен к отражению"
    diff = (a[:, n_geo:] - b[:, n_geo:]).abs().max()
    assert diff > 0.1, f"ориентационный блок не реагирует на отражение (max diff {diff})"
