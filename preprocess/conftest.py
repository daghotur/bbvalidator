import pytest
import torch

from preprocess.foldability_features import FoldabilityProxies
from preprocess.geometry_features import BackboneGeometryExtractor


@pytest.fixture(scope="module")
def device():
    # Тесты лёгкие — CPU, чтобы не зависеть от наличия GPU
    return torch.device("cpu")


@pytest.fixture
def extractor(device):
    return BackboneGeometryExtractor().to(device)


@pytest.fixture
def proxy(device):
    return FoldabilityProxies(
        contact_threshold=8.0, seq_sep=3, pca_components=16, fragment_size=9
    ).to(device)
