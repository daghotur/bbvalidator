import time
import math
import torch
from preprocess.geometry_features import BackboneGeometryExtractor


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------


def _place_atom(a, b, c, bond_len, bond_angle, torsion):
    """
    NeRF: разместить атом D с заданными bond_len(C-D), bond_angle(B-C-D)
    и torsion(A-B-C-D) относительно трёх предыдущих атомов A, B, C.
    Все аргументы — torch.Tensor формы [3].
    """
    # Единичные векторы
    bc = c - b
    bc = bc / torch.linalg.norm(bc)
    ab = b - a
    n = torch.linalg.cross(ab, bc)
    n = n / torch.linalg.norm(n)
    m = torch.linalg.cross(n, bc)

    # Локальные координаты D
    d_x = -bond_len * math.cos(bond_angle)
    d_y = bond_len * math.cos(torsion) * math.sin(bond_angle)
    d_z = bond_len * math.sin(torsion) * math.sin(bond_angle)

    d = c + d_x * bc + d_y * m + d_z * n
    return d


def _make_helix_coords(B: int, N: int, device: torch.device, noise: float = 0.05):
    """
    Идеальная α-спираль, построенная через NeRF с реалистичными
    длинами связей, валентными углами и торсионами (phi=-57°, psi=-47°, omega=180°).
    Возвращает тензор [B, N, 3, 3] (атомы: N, CA, C).
    """
    # Длины связей (Å)
    L_N_CA = 1.458
    L_CA_C = 1.525
    L_C_N = 1.329

    # Валентные углы (радианы) — это углы между связями (π - угол изгиба в NeRF)
    A_N_CA_C = math.radians(111.0)
    A_CA_C_N = math.radians(116.0)
    A_C_N_CA = math.radians(122.0)

    # Торсионы α-спирали
    PHI = math.radians(-57.0)  # C(i-1) - N(i)  - CA(i) - C(i)
    PSI = math.radians(-47.0)  # N(i)   - CA(i) - C(i)  - N(i+1)
    OMG = math.radians(180.0)  # CA(i)  - C(i)  - N(i+1) - CA(i+1)

    # Стартовые три атома: N0, CA0, C0
    atoms = []
    N0 = torch.tensor([0.0, 0.0, 0.0], device=device)
    CA0 = torch.tensor([L_N_CA, 0.0, 0.0], device=device)
    # C0 в плоскости xy с правильным углом N-CA-C
    C0 = CA0 + torch.tensor(
        [L_CA_C * math.cos(math.pi - A_N_CA_C),
         L_CA_C * math.sin(math.pi - A_N_CA_C),
         0.0],
        device=device,
    )
    atoms.extend([N0, CA0, C0])

    # Дальнейшие атомы по NeRF.
    # Для каждого следующего остатка i: ставим N(i), CA(i), C(i)
    # с торсионами psi(i-1), omega(i-1), phi(i) соответственно.
    for i in range(1, N):
        a, b, c = atoms[-3], atoms[-2], atoms[-1]  # ... N(i-1), CA(i-1), C(i-1)

        # N(i): торсион psi(i-1) вокруг связи CA(i-1)-C(i-1), угол CA-C-N
        N_i = _place_atom(a, b, c, L_C_N, math.pi - A_CA_C_N, PSI)
        atoms.append(N_i)

        # CA(i): торсион omega вокруг связи C(i-1)-N(i), угол C-N-CA
        a, b, c = atoms[-3], atoms[-2], atoms[-1]  # CA(i-1), C(i-1), N(i)
        CA_i = _place_atom(a, b, c, L_N_CA, math.pi - A_C_N_CA, OMG)
        atoms.append(CA_i)

        # C(i): торсион phi(i) вокруг связи N(i)-CA(i), угол N-CA-C
        a, b, c = atoms[-3], atoms[-2], atoms[-1]  # C(i-1), N(i), CA(i)
        C_i = _place_atom(a, b, c, L_CA_C, math.pi - A_N_CA_C, PHI)
        atoms.append(C_i)

    # Собираем в [N, 3 (atoms N,CA,C), 3 (xyz)]
    flat = torch.stack(atoms, dim=0)  # [N*3, 3]
    per_res = flat.view(N, 3, 3)

    coords = per_res.unsqueeze(0).expand(B, -1, -1, -1).clone()
    if noise > 0:
        coords += torch.randn_like(coords) * noise
    return coords


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------


def test_output_shapes(extractor, device):
    """Проверка форм всех выходных тензоров."""
    B, N = 8, 50
    coords = _make_helix_coords(B, N, device)

    feats = extractor(coords)
    packed = extractor.pack_for_mlp(feats)

    # Скалярные признаки
    for key in ("phi", "psi", "omega", "ca_dist", "hbond_count", "clash_count"):
        assert feats[key].shape == (
            B,
            N,
        ), f"Неверная форма для '{key}': {feats[key].shape}"

    # Булевы маски
    for key in (
            "phi_mask",
            "psi_mask",
            "omega_mask",
            "ca_mask",
            "global_mask",
            "ram_outliers",
    ):
        assert feats[key].shape == (
            B,
            N,
        ), f"Неверная форма для '{key}': {feats[key].shape}"

    assert packed.shape == (
        B,
        N,
        10,
    ), f"pack_for_mlp: ожидали (B, N, 10), получили {packed.shape}"
    print("  [OK] output shapes")


def test_mask_correctness(extractor, device):
    """
    Проверяем правильность масок после исправления бага:
      phi   определён для остатков 1..N-1  → mask[0]=False, mask[-1]=True
      psi   определён для остатков 0..N-2  → mask[0]=True,  mask[-1]=False
      omega определён для остатков 1..N-1  → mask[0]=False, mask[-1]=True
      ca_dist определён для остатков 0..N-2 → mask[-1]=False
    """
    B, N = 4, 20
    coords = _make_helix_coords(B, N, device, noise=0.0)
    feats = extractor(coords)

    # --- phi ---
    assert not feats["phi_mask"][
        :, 0
    ].any(), "phi_mask[:,0] должен быть False (нет C предыдущего)"
    assert feats["phi_mask"][:, 1:].all(), "phi_mask[:,1:] должен быть True"
    # Ключевой тест: до исправления phi_mask[:, -1] ошибочно был False
    assert feats["phi_mask"][
        :, -1
    ].all(), "phi_mask[:,-1] должен быть True (баг-регрессия)"

    # --- psi ---
    assert feats["psi_mask"][:, 0].all(), "psi_mask[:,0] должен быть True"
    assert not feats["psi_mask"][
        :, -1
    ].any(), "psi_mask[:,-1] должен быть False (нет N следующего)"
    assert feats["psi_mask"][:, :-1].all()

    # --- omega ---
    assert not feats["omega_mask"][:, 0].any(), "omega_mask[:,0] должен быть False"
    assert feats["omega_mask"][:, 1:].all(), "omega_mask[:,1:] должен быть True"
    assert feats["omega_mask"][
        :, -1
    ].all(), "omega_mask[:,-1] должен быть True (баг-регрессия)"

    # --- ca_dist ---
    assert feats["ca_mask"][:, :-1].all()
    assert not feats["ca_mask"][
        :, -1
    ].any(), "ca_mask[:,-1] должен быть False (нет следующего CA)"

    print("  [OK] mask correctness (включая регрессию phi/omega)")


def test_padding_zeroed(extractor, device):
    """Паддинговые позиции должны давать нули после pack_for_mlp."""
    B, N, valid_len = 4, 30, 20
    coords = _make_helix_coords(B, N, device)

    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    mask[:, :valid_len] = True

    feats = extractor(coords, mask)
    packed = extractor.pack_for_mlp(feats)

    pad_region = packed[:, valid_len:]
    assert (pad_region == 0).all(), (
        f"Паддинговые позиции [{valid_len}:] должны быть нулями, "
        f"max_abs={pad_region.abs().max().item():.6f}"
    )
    print("  [OK] padded positions are zeroed")


def test_ramachandran_helix(extractor, device):
    """
    Для идеальной α-спирали большинство остатков должны находиться
    в допустимой области Рамачандрана (мало аутлайеров).
    """
    B, N = 2, 100
    # Без шума, чистая спираль
    coords = _make_helix_coords(B, N, device, noise=0.0)
    feats = extractor(coords)

    # Исключаем граничные остатки (не имеют phi или psi)
    interior_mask = feats["phi_mask"] & feats["psi_mask"]
    interior_total = interior_mask.sum().item()
    interior_outliers = (feats["ram_outliers"] & interior_mask).sum().item()

    outlier_rate = interior_outliers / max(interior_total, 1)
    assert outlier_rate < 0.5, (
        f"Слишком много аутлайеров для спирали: {outlier_rate:.1%} "
        f"(проверь логику ram_outliers)"
    )
    print(f"  [OK] Ramachandran helix outlier rate: {outlier_rate:.1%}")


def test_ca_distances(extractor, device):
    """Расстояния Cα–Cα в спирали должны быть близки к 3.8 Å."""
    B, N = 2, 50
    coords = _make_helix_coords(B, N, device, noise=0.0)
    feats = extractor(coords)

    valid_ca = feats["ca_dist"][feats["ca_mask"]]
    mean_dist = valid_ca.mean().item()
    # Спираль с параметрами из _make_helix_coords: шаг ~3.5–4.5 Å
    assert (
            2.5 < mean_dist < 6.0
    ), f"Среднее расстояние CA-CA = {mean_dist:.2f} Å — за пределами ожидаемого"
    print(f"  [OK] CA-CA mean distance: {mean_dist:.3f} Å")


# ---------------------------------------------------------------------------
# Бенчмарк
# ---------------------------------------------------------------------------


def benchmark(extractor, device, B: int = 4096, N: int = 200, reps: int = 10):
    coords = _make_helix_coords(B, N, device)

    # Прогрев (особенно важен для CUDA JIT-компиляции и аллокаций)
    with torch.inference_mode():
        for _ in range(3):
            extractor(coords)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(reps):
            feats = extractor(coords)
            packed = extractor.pack_for_mlp(feats)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / reps
    throughput = B / elapsed
    latency_ms = (elapsed / B) * 1e3

    print(f"\n  Батч: {B} backbones | N={N} | {reps} прогонов")
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
    # FIX: убран параметр device= — он был удалён из конструктора BackboneGeometryExtractor
    extractor = BackboneGeometryExtractor().to(device)

    print(f"\n=== BackboneGeometryExtractor | device={device} ===\n")

    test_output_shapes(extractor, device)
    test_mask_correctness(extractor, device)
    test_padding_zeroed(extractor, device)
    test_ramachandran_helix(extractor, device)
    test_ca_distances(extractor, device)

    print("\n--- Benchmark ---")
    benchmark(extractor, device)
    print("\nВсе тесты пройдены.")


if __name__ == "__main__":
    run_all()
