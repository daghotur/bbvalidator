import time
import torch
from preprocess.designability_features import DesignabilityProxies


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------

def _make_random_walk(B: int, N: int, device: torch.device, seed: int = 42):
    """
    Синтетические Cα-координаты: масштабированный random walk ~3.8 Å.
    Центрируем каждый белок, чтобы поведение relative_burial было предсказуемым.
    """
    torch.manual_seed(seed)
    coords = torch.randn(B, N, 3, device=device).cumsum(dim=1) * 3.8
    coords -= coords.mean(dim=1, keepdim=True)
    return coords


def _make_proxy(device: torch.device, **kwargs) -> DesignabilityProxies:
    return DesignabilityProxies(
        contact_threshold=8.0, seq_sep=3, pca_components=16, fragment_size=9, **kwargs
    ).to(device)


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------

def test_output_shapes(proxy, device):
    """Проверка форм всех выходных тензоров."""
    B, N = 8, 50
    coords = _make_random_walk(B, N, device)
    mask   = torch.ones(B, N, dtype=torch.bool, device=device)

    feats  = proxy(coords, mask)
    packed = proxy.pack_for_mlp(feats)

    for key in ("packing_r6", "packing_r8", "packing_r10",
                "relative_burial", "local_bending"):
        assert feats[key].shape == (B, N), f"'{key}': ожидали ({B},{N}), получили {feats[key].shape}"

    assert feats["pca_projection"].shape == (B, N, 16), (
        f"pca_projection: {feats['pca_projection'].shape}"
    )
    assert packed.shape == (B, N, 21), f"pack_for_mlp: {packed.shape}"  # 5 scalar + 16 PCA
    print("  [OK] output shapes")


def test_local_bending_boundaries(proxy, device):
    """
    local_bending вычисляется как ||CA[i+4] - CA[i-4]||.
    После pad(2, 2) первые 2 и последние 2 позиции должны быть нулями.
    """
    B, N = 4, 30
    coords = _make_random_walk(B, N, device)
    mask   = torch.ones(B, N, dtype=torch.bool, device=device)

    feats = proxy(coords, mask)
    lb    = feats["local_bending"]

    assert (lb[:, :2] == 0).all(),  "local_bending[:,0:2] должен быть 0 (паддинг слева)"
    assert (lb[:, -2:] == 0).all(), "local_bending[:,-2:] должен быть 0 (паддинг справа)"
    assert (lb[:, 2:-2] > 0).all(), "local_bending во внутренних позициях должен быть > 0"
    print("  [OK] local_bending boundaries")


def test_padding_zeroed(proxy, device):
    """Паддинговые позиции (mask=False) должны давать нули в packed-выводе."""
    B, N, valid_len = 6, 40, 25
    coords = _make_random_walk(B, N, device)
    mask   = torch.zeros(B, N, dtype=torch.bool, device=device)
    mask[:, :valid_len] = True

    feats  = proxy(coords, mask)
    packed = proxy.pack_for_mlp(feats)

    pad_region = packed[:, valid_len:]
    assert (pad_region == 0).all(), (
        f"Паддинговые позиции [{valid_len}:] должны быть нулями, "
        f"max_abs={pad_region.abs().max().item():.6f}"
    )
    print("  [OK] padded positions are zeroed")


def test_dist_mat_passthrough(proxy, device):
    """
    Передача предвычисленного dist_mat должна давать идентичный результат
    и экономить повторный вызов torch.cdist.
    """
    B, N = 4, 30
    coords   = _make_random_walk(B, N, device)
    mask     = torch.ones(B, N, dtype=torch.bool, device=device)
    dist_mat = torch.cdist(coords, coords)

    feats_auto = proxy(coords, mask)
    feats_pre  = proxy(coords, mask, dist_mat=dist_mat)

    for key in ("packing_r6", "packing_r8", "packing_r10",
                "relative_burial", "pca_projection"):
        assert torch.allclose(feats_auto[key], feats_pre[key], atol=1e-5), (
            f"Расхождение при передаче dist_mat для '{key}'"
        )
    print("  [OK] dist_mat passthrough gives identical results")


def test_freeze_pca(proxy, device):
    """freeze_pca() должна полностью отключать градиенты PCA-весов."""
    # До заморозки
    assert proxy.pca_proj.weight.requires_grad, \
        "До freeze_pca() requires_grad должен быть True"

    proxy.freeze_pca()
    assert not proxy.pca_proj.weight.requires_grad, \
        "После freeze_pca() requires_grad должен быть False"

    proxy.unfreeze_pca()
    assert proxy.pca_proj.weight.requires_grad, \
        "После unfreeze_pca() requires_grad должен быть True"

    # Дополнительно: проверяем, что градиент не течёт через замороженный слой
    proxy.freeze_pca()
    coords = _make_random_walk(2, 20, device)
    mask   = torch.ones(2, 20, dtype=torch.bool, device=device)

    # Чтобы граф autograd вообще существовал при замороженном pca_proj
    # (в реальной сети градиент приходит от вышестоящих обучаемых слоёв),
    # помечаем входные координаты как требующие градиента.
    coords.requires_grad_(True)

    # Запускаем без inference_mode, чтобы граф строился
    feats  = proxy(coords, mask)
    packed = proxy.pack_for_mlp(feats)
    loss   = packed.sum()
    loss.backward()

    assert proxy.pca_proj.weight.grad is None, \
        "Градиент не должен вычисляться для замороженного pca_proj"

    proxy.unfreeze_pca()
    print("  [OK] freeze_pca / unfreeze_pca")


def test_contact_density_sanity(proxy, device):
    """
    Плотность контактов должна быть неубывающей по радиусу:
    r6 <= r8 <= r10 для всех остатков.
    """
    B, N = 4, 60
    coords = _make_random_walk(B, N, device)
    mask   = torch.ones(B, N, dtype=torch.bool, device=device)
    feats  = proxy(coords, mask)

    assert (feats["packing_r6"] <= feats["packing_r8"] + 1e-5).all(), \
        "packing_r6 должен быть <= packing_r8"
    assert (feats["packing_r8"] <= feats["packing_r10"] + 1e-5).all(), \
        "packing_r8 должен быть <= packing_r10"
    print("  [OK] contact density r6 <= r8 <= r10")


def test_no_mask_defaults(proxy, device):
    """Вызов без маски эквивалентен маске из единиц."""
    B, N = 2, 20
    coords    = _make_random_walk(B, N, device)
    full_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    feats_no_mask   = proxy(coords)
    feats_full_mask = proxy(coords, full_mask)

    assert torch.allclose(
        feats_no_mask["packing_r8"], feats_full_mask["packing_r8"], atol=1e-5
    ), "Результат без маски должен совпадать с маской из единиц"
    print("  [OK] None mask == all-ones mask")


# ---------------------------------------------------------------------------
# Бенчмарк
# ---------------------------------------------------------------------------

def benchmark(proxy, device, B: int = 4096, N: int = 200, reps: int = 10):
    coords = _make_random_walk(B, N, device)
    mask   = torch.zeros(B, N, dtype=torch.bool, device=device)
    mask[:, :180] = True  # 20 паддинг-позиций

    # Прогрев (важен для CUDA: компиляция ядер, аллокации памяти)
    with torch.inference_mode():
        for _ in range(3):
            proxy(coords, mask)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(reps):
            DesignabilityProxies.pack_for_mlp(proxy(coords, mask))

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed    = (time.perf_counter() - start) / reps
    throughput = B / elapsed
    latency_ms = (elapsed / B) * 1e3

    print(f"\n  Батч: {B} backbones | N={N} | valid={180} | {reps} прогонов")
    print(f"  Время/прогон: {elapsed:.4f} с | Latency: {latency_ms:.3f} мс/шт")
    print(f"  Throughput:   {throughput:.0f} шт/с  (~{throughput * 3600 / 1e3:.0f}k/ч)")
    if device.type == "cuda":
        print(f"  VRAM peak:    {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    else:
        print("  VRAM: CUDA недоступна, используется CPU")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proxy  = _make_proxy(device)

    print(f"\n=== DesignabilityProxies | device={device} ===\n")

    test_output_shapes(proxy, device)
    test_local_bending_boundaries(proxy, device)
    test_padding_zeroed(proxy, device)
    test_dist_mat_passthrough(proxy, device)
    test_freeze_pca(proxy, device)
    test_contact_density_sanity(proxy, device)
    test_no_mask_defaults(proxy, device)

    print("\n--- Benchmark ---")
    benchmark(proxy, device)
    print("\nВсе тесты пройдены.")


if __name__ == "__main__":
    run_all()