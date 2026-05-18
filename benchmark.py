import argparse
import time
import torch
import numpy as np
import warnings
from inference import build_model, parse_pdb_to_backbone, center_coords
from model.heads_loss import predict_with_uncertainty

warnings.filterwarnings("ignore", message=".*elements were guessed from atom name.*")


def benchmark_model(pdb_path: str, ckpt_path: str, mc_runs: int, iterations: int, device: torch.device):
    model = build_model(ckpt_path, device)

    print(f"Подготовка данных: {pdb_path}...")
    coords_np = parse_pdb_to_backbone(pdb_path)
    coords_np = center_coords(coords_np)
    length = len(coords_np)

    coords_ts = torch.from_numpy(coords_np).unsqueeze(0).to(device)
    mask_ts = torch.ones((1, length), dtype=torch.bool, device=device)

    print(f"Прогрев 5 итераций...")
    for _ in range(5):
        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16) if device.type == 'cuda' else torch.no_grad():
            _ = predict_with_uncertainty(model, coords_ts, mask_ts, mc_runs=mc_runs)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f"Старт бенчмарка ({iterations} итераций, MC runs: {mc_runs})...")

    times = []

    for i in range(iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16) if device.type == 'cuda' else torch.no_grad():
            _ = predict_with_uncertainty(model, coords_ts, mask_ts, mc_runs=mc_runs)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        times.append((end_time - start_time) * 1000)

    avg_time = np.mean(times)
    p90_time = np.percentile(times, 90)
    fps = 1000.0 / avg_time

    print("-" * 50)
    print(f"Результаты для белка длиной {length} аминокислот:")
    print(f"Среднее время (Latency) : {avg_time:.2f} мс")
    print(f"90-й перцентиль         : {p90_time:.2f} мс")
    print(f"Пропускная способность  : {fps:.2f} белков в секунду (FPS)")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Бенчмарк скорости ProteinScoreModel")
    parser.add_argument("-i", "--input", required=True, help="Путь к ОДНОМУ .pdb файлу для теста")
    parser.add_argument("-c", "--ckpt", default="checkpoints/best_model.pth", help="Путь к весам")
    parser.add_argument("-m", "--mc_runs", type=int, default=16, help="Количество MC-Dropout проходов")
    parser.add_argument("-n", "--iterations", type=int, default=50, help="Количество итераций теста")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_model(args.input, args.ckpt, args.mc_runs, args.iterations, device)


if __name__ == "__main__":
    main()
