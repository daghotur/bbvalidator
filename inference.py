import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import contextlib
import warnings

from preprocess.biophys_frontend import BiophysicalFrontend
from preprocess.fit_pca import load_pca_into_frontend
from model.encoder import HybridProteinEncoder
from model.heads_loss import (
    MultiHeadAttentionPooling,
    PerResidueHead,
    ProteinMultiTaskHeads,
    ProteinScoreModel,
    predict_with_uncertainty
)

warnings.filterwarnings("ignore", message=".*elements were guessed from atom name.*")


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


def soft_checkpoint_marker(checkpoint: dict) -> str | None:
    """Метка soft-дообучения в чекпоинте, если она есть.

    training/soft.py кладёт target="log1p(scRMSD)", training/soft_multitask.py —
    recipe с native-якорем. По ним отличается семантика головы fold.
    """
    if not isinstance(checkpoint, dict):
        return None
    for key in ("target", "recipe"):
        value = checkpoint.get(key)
        if isinstance(value, str):
            return f"{key}={value}"
    return None


def expand_linear_inputs(
    state_dict: dict, model: nn.Module, pair_init: str = "zero"
) -> dict:
    """Дополняет входы линейных слоёв, у которых выросла размерность входа.

    Повод — расширение парного канала 20 → 29: к признакам рёбер добавлены
    ориентационные (набор trRosetta, docs/05 раздел 5.8). Новые входы
    дописываются В КОНЕЦ конкатенации, поэтому старые веса остаются на своих
    местах и чекпоинты остаются загружаемыми.

    Слои не перечисляются поимённо: у гибрида это `encoder.pair_embed.0`, у
    GPS-базлайна — `encoder.gps_layers.*.conv.lin_edge`, и список пришлось бы
    поддерживать вручную (из-за чего базлайны и перестали загружаться).
    Признак случая объективен: двумерный вес, число выходов совпало, а входов
    в чекпоинте меньше. Всё остальное несовпадение форм остаётся ошибкой
    загрузки — её поднимет сам load_state_dict.

    pair_init="zero"   — новые входы зануляются: модель сразу после загрузки
                         считает ровно то же, что и раньше. Режим для инференса
                         и воспроизведения старых чисел.
    pair_init="scaled" — новые входы инициализируются в масштабе слоя. Режим
                         для дообучения: при нулевом старте поверх уже
                         сошедшейся сети новые входы остаются мёртвыми (замер:
                         после 45 эпох норма ориентационных весов была 0.15
                         против 5.97 у остальных, зеркальный тест не сдвинулся),
                         потому что давления их включать нет.
    """
    state_dict = dict(state_dict)
    for key, param in model.state_dict().items():
        if key not in state_dict or not key.endswith(".weight"):
            continue
        have, want = state_dict[key].shape, param.shape
        if len(have) != 2 or len(want) != 2:
            continue
        if have[0] != want[0] or have[1] >= want[1]:
            continue
        if pair_init == "zero":
            padded = torch.zeros(want, dtype=state_dict[key].dtype)
            note = "нулями (поведение не меняется)"
        elif pair_init == "scaled":
            bound = 1.0 / (want[1] ** 0.5)
            padded = torch.empty(want, dtype=state_dict[key].dtype).uniform_(-bound, bound)
            note = f"в масштабе слоя ±{bound:.3f} (для дообучения)"
        else:
            raise ValueError(
                f"pair_init: ожидается 'zero' или 'scaled', получено {pair_init!r}"
            )
        padded[:, : have[1]] = state_dict[key]
        state_dict[key] = padded
        print(f"{key}: вход расширен {have[1]} → {want[1]}, новые входы {note}.")
    return state_dict


def build_model(
    checkpoint_path: str,
    device: torch.device,
    pca_path: str = "dataset/pca_components.pth",
    per_residue: bool = False,
    pair_init: str = "zero",
) -> ProteinScoreModel:
    d_model = 192
    frontend = BiophysicalFrontend(use_no_grad=True)
    encoder = HybridProteinEncoder(
        node_in_dim=31,
        # берём у фронтенда, чтобы размерности не разъезжались при правках признаков
        pair_in_dim=frontend.pair_builder.feature_dim,
        d_model=d_model,
        pair_dim=64,
        num_graph_layers=2,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.15
    )
    pooler = MultiHeadAttentionPooling(d_model=d_model, num_heads=4)
    heads = ProteinMultiTaskHeads(d_model=d_model, dropout=0.15)
    per_residue_head = PerResidueHead(d_model=d_model, dropout=0.15) if per_residue else None

    model = ProteinScoreModel(frontend, encoder, pooler, heads, per_residue_head)

    print(f"Загрузка весов из {checkpoint_path}...")
    # weights_only=True защищает от исполнения произвольного кода при загрузке
    # недоверенного .pth (pickle RCE). Чекпоинт содержит только тензоры и скаляры.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Обработка ключа model_state_dict (как мы сохраняли в training/hybrid.py)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = expand_linear_inputs(state_dict, model, pair_init)
    try:
        if per_residue and not any(k.startswith("per_residue_head.") for k in state_dict):
            # Дообучение поверх чекпоинта без по-остаточной головы: она новая,
            # инициализируется с нуля. Всё остальное обязано совпасть.
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            missing = [k for k in missing if not k.startswith("per_residue_head.")]
            if missing or unexpected:
                raise RuntimeError(f"несовпадение ключей: {missing[:5]} / {unexpected[:5]}")
            print("По-остаточная голова инициализирована с нуля (в чекпоинте её нет).")
        else:
            model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            "Чекпоинт несовместим с текущей архитектурой. Чекпоинты до "
            "пересборки 2026-08 не подойдут: убрана hbond-голова, добавлен "
            "PCA-буфер frag_mean. Переобучите модель (training/hybrid.py)."
        ) from e

    # PCA фронтенда: чекпоинт уже содержит фит-веса, но файл применяем
    # повторно, чтобы гарантировать согласованность и заморозку.
    if os.path.exists(pca_path):
        load_pca_into_frontend(model.frontend, pca_path)
    else:
        print(f"ВНИМАНИЕ: {pca_path} не найден — используются PCA-веса из чекпоинта.")

    model.to(device)
    model.eval()
    return model


def _autocast_ctx(device: torch.device):
    """bfloat16-автокаст на CUDA, иначе — no-op (CPU не выигрывает от bf16)."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def process_single_pdb(pdb_path: str, model: ProteinScoreModel, device: torch.device, mc_runs: int):
    try:
        # 1. Парсинг и препроцессинг
        coords_np = parse_pdb_to_backbone(pdb_path)
        length = len(coords_np)
        coords_np = center_coords(coords_np)  # КРИТИЧНО!

        # 2. Подготовка тензоров [B, N, 3, 3], где B=1
        coords_ts = torch.from_numpy(coords_np).unsqueeze(0).to(device)
        mask_ts = torch.ones((1, length), dtype=torch.bool, device=device)

        # 3. MC-Dropout Инференс (predict_with_uncertainty уже под @torch.no_grad)
        with _autocast_ctx(device):
            results = predict_with_uncertainty(model, coords_ts, mask_ts, mc_runs=mc_runs)

        p_foldable = results["p_foldable"].item()
        uncertainty = results["uncertainty"].item()

        # Оценка дополнительных голов (просто один проход для логов)
        with torch.no_grad(), _autocast_ctx(device):
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

    # Колонки ниже (P(Fold), Uncert., Pred RMSD) читают fold_logit как логит
    # вероятности. У soft-чекпоинтов это регрессия log1p(scRMSD): sigmoid от
    # неё — не вероятность, и «✅ 0.99» означал бы ровно обратное.
    marker = soft_checkpoint_marker(
        torch.load(args.ckpt, map_location="cpu", weights_only=True)
    )
    if marker:
        raise SystemExit(
            f"{args.ckpt} — soft-чекпоинт ({marker}): голова fold предсказывает "
            "log1p(scRMSD), а не вероятность фолдинга. Для таких весов есть "
            "filter_designability.py — он печатает предсказанный scRMSD в Å."
        )

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
