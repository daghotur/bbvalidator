import os
import argparse
import numpy as np
import torch
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import contextlib
import warnings

warnings.filterwarnings("ignore", message=".*elements were guessed from atom name.*")

from preprocess.biophys_frontend import BiophysicalFrontend
from model.encoder import HybridProteinEncoder
from model.heads_loss import (
    MultiHeadAttentionPooling,
    ProteinMultiTaskHeads,
    ProteinScoreModel,
    predict_with_uncertainty
)


def center_coords(coords: np.ndarray) -> np.ndarray:
    centroid = coords.mean(axis=(0, 1), keepdims=True)
    return coords - centroid


def parse_pdb_to_backbone(pdb_path: str) -> np.ndarray:
    """
    Извлекает координаты [N, CA, C] из PDB файла.
    Возвращает numpy массив формы [L, 3, 3].
    """
    pdb_file = pdb.PDBFile.read(pdb_path)
    # Берем первую модель из PDB
    structure = pdb.get_structure(pdb_file, model=1)

    # Оставляем только стандартные аминокислоты
    chain_struc = structure[struc.filter_amino_acids(structure)]

    # Извлекаем атомы (аналогично build_positive_dataset.py)
    residue_starts = struc.get_residue_starts(chain_struc)
    coords_list = []

    for i, start in enumerate(residue_starts):
        end = residue_starts[i + 1] if i + 1 < len(residue_starts) else len(chain_struc)
        res_atoms = chain_struc[start:end]
        names = res_atoms.atom_name

        n_idx = np.where(names == "N")[0]
        ca_idx = np.where(names == "CA")[0]
        c_idx = np.where(names == "C")[0]

        if len(n_idx) == 1 and len(ca_idx) == 1 and len(c_idx) == 1:
            coords_list.append([
                res_atoms.coord[n_idx[0]],
                res_atoms.coord[ca_idx[0]],
                res_atoms.coord[c_idx[0]],
            ])

    if not coords_list:
        raise ValueError(f"Не удалось извлечь backbone из {pdb_path}")

    coords = np.array(coords_list, dtype=np.float32)
    return coords


def build_model(checkpoint_path: str, device: torch.device) -> ProteinScoreModel:
    d_model = 192
    frontend = BiophysicalFrontend(use_no_grad=True)
    encoder = HybridProteinEncoder(
        node_in_dim=31,
        pair_in_dim=20,
        d_model=d_model,
        pair_dim=64,
        num_graph_layers=2,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.15
    )
    pooler = MultiHeadAttentionPooling(d_model=d_model, num_heads=4)
    heads = ProteinMultiTaskHeads(d_model=d_model, dropout=0.15)

    model = ProteinScoreModel(frontend, encoder, pooler, heads)

    print(f"Загрузка весов из {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Обработка ключа model_state_dict (как мы сохраняли в train_model.py)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def process_single_pdb(pdb_path: str, model: ProteinScoreModel, device: torch.device, mc_runs: int):
    try:
        # 1. Парсинг и препроцессинг
        coords_np = parse_pdb_to_backbone(pdb_path)
        length = len(coords_np)
        coords_np = center_coords(coords_np)  # КРИТИЧНО!

        # 2. Подготовка тензоров [B, N, 3, 3], где B=1
        coords_ts = torch.from_numpy(coords_np).unsqueeze(0).to(device)
        mask_ts = torch.ones((1, length), dtype=torch.bool, device=device)

        # 3. MC-Dropout Инференс
        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16) if device.type == 'cuda' else torch.no_grad():
            results = predict_with_uncertainty(model, coords_ts, mask_ts, mc_runs=mc_runs)

        p_foldable = results["p_foldable"].item()
        uncertainty = results["uncertainty"].item()

        # Оценка дополнительных голов (просто один проход для логов)
        with torch.no_grad(), torch.autocast(device_type=device.type,
                                             dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext():
            single_preds = model(coords_ts, mask_ts)
            rmsd_pred = torch.expm1(single_preds["rmsd"]).item()
            rmsd_pred = max(0.0, rmsd_pred)
            fail_mode = torch.argmax(single_preds["failure_mode"], dim=-1).item()

        return {
            "p_foldable": p_foldable,
            "uncertainty": uncertainty,
            "rmsd_pred": rmsd_pred,
            "fail_mode_idx": fail_mode,
            "length": length
        }

    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Инференс ProteinScoreModel для PDB файлов")
    parser.add_argument("-i", "--input", required=True, help="Путь к .pdb файлу или директории с файлами")
    parser.add_argument("-c", "--ckpt", default="checkpoints/best_model.pth", help="Путь к файлу весов (.pth)")
    parser.add_argument("-m", "--mc_runs", type=int, default=16,
                        help="Количество проходов MC-Dropout (по умолчанию 16)")
    parser.add_argument("--cpu", action="store_true", help="Форсировать использование CPU")
    args = parser.parse_args()

    # Устройство
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Инференс на устройстве: {device}")

    # Модель
    if not os.path.exists(args.ckpt):
        print(f"Ошибка: Файл весов {args.ckpt} не найден.")
        return
    model = build_model(args.ckpt, device)

    # Поиск файлов
    target_files = []
    if os.path.isfile(args.input) and args.input.endswith(".pdb"):
        target_files.append(args.input)
    elif os.path.isdir(args.input):
        target_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".pdb")]

    if not target_files:
        print(f"Ошибка: Не найдено .pdb файлов по пути {args.input}")
        return

    print(f"\nЗапуск инференса для {len(target_files)} файлов (MC runs: {args.mc_runs})...\n")
    print("-" * 70)
    print(f"{'Файл':<25} | {'P(Fold)':<8} | {'Uncert.':<8} | {'Pred RMSD':<9} | {'Len':<4}")
    print("-" * 70)

    for pdb_file in target_files:
        res = process_single_pdb(pdb_file, model, device, args.mc_runs)

        filename = os.path.basename(pdb_file)[:23]
        if "error" in res:
            print(f"{filename:<25} | ОШИБКА: {res['error']}")
        else:
            pf = res['p_foldable']
            unc = res['uncertainty']
            rmsd = res['rmsd_pred']
            L = res['length']

            # Визуальный индикатор уверенности: ✅ - ок, ⚠️ - сомнительно, ❌ - плохой фолд
            icon = "✅" if pf > 0.8 else ("⚠️" if pf > 0.4 else "❌")

            print(f"{filename:<25} | {pf:.4f} {icon} | {unc:.5f}  | {rmsd:.3f}     | {L:<4}")


if __name__ == "__main__":
    main()
