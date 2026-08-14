"""
common/motifbench.py
--------------------
Данные MotifBench: где лежат структуры генераторов, где — их self-consistency
оценки, и как из восьми per-sequence RMSD получается метка скаффолда.

Единственная точка чтения `*_eval_results.csv` во всём проекте. До сведения
парсеров сюда их было пять штук в разных файлах, и один из них (обогащение)
остался на старом агрегаторе после смены метки 13.08.2026 — то есть считал
свои числа против другой истины, чем весь остальной пайплайн.
"""

import glob
import os

import numpy as np
import pandas as pd

# Структуры генераторов
GENERATOR_DIRS = {
    "RFdiffusion": "data/ood/rfdiffusion",
    "RFdiffusion-AA": "data/ood/rfdiffusion_aa",
    "ODesign-Rigid": "data/ood/odesign_rigid",
    "GPDL": "data/ood/gpdl",
    "EvoDiff": "data/ood/evodiff",
}

# Их self-consistency оценки (рефолды + RMSD по каждой последовательности)
EVAL_SOURCES = {
    "RFdiffusion": "data/ood/eval_rfdiffusion",
    "RFdiffusion-AA": "data/ood/eval_rfdiffusion_aa",
    "ODesign-Rigid": "data/ood/eval_odesign_rigid",
    "GPDL": "data/ood/eval_gpdl",
    "EvoDiff": "data/ood/eval_evodiff",
}

# Рефолдеры: ESMFold — «свой» оракул MotifBench, AF2 — независимая проверка
ORACLE_CSV = {"ESMFold": "esm_eval_results.csv", "AF2": "af2_eval_results.csv"}

# Протокол hold-out для всех дообучений на мягкую метку: два генератора не
# участвуют в обучении вообще, и финальные числа отчитываются по ним.
TRAIN_GENERATORS = ["RFdiffusion", "RFdiffusion-AA", "EvoDiff"]
HOLDOUT_GENERATORS = ["ODesign-Rigid", "GPDL"]

# Доля мотивов, отложенных под выбор эпохи. Делить скаффолды случайно нельзя:
# мотивы у train и val тогда общие, и val меряет обобщение на новые скаффолды
# знакомой мишени, а не на новую мишень. Мотивы одни и те же у всех
# генераторов, поэтому разбиение глобальное и одинаковое во всех скриптах.
VAL_MOTIF_FRACTION = 0.2
MOTIF_SPLIT_SEED = 42


def motif_role(generator: str, motif: str, val_motifs: set[str]) -> str:
    """Что означает число, посчитанное на этом (генераторе, мотиве).

    "in-sample" — генератор и мотив участвовали в дообучении;
    "val-motif" — генератор обучающий, но мотив отложен под выбор эпохи;
    "unseen"    — генератор целиком в холдауте, модель его не видела.

    Нужно затем, чтобы отчёты анализов не смешивали обобщение с запоминанием:
    три генератора из пяти участвуют в дообучении, и их числа — не OOD.
    """
    if generator in HOLDOUT_GENERATORS:
        return "unseen"
    return "val-motif" if motif in val_motifs else "in-sample"


def motif_split(
    motifs, val_fraction: float = VAL_MOTIF_FRACTION, seed: int = MOTIF_SPLIT_SEED
) -> tuple[set[str], set[str]]:
    """(train-мотивы, val-мотивы) — детерминированно от имён мотивов.

    Состояние никуда не сохраняется: любой скрипт и любой анализ восстановит
    то же разбиение из того же списка мотивов.
    """
    ordered = sorted(set(motifs))
    rng = np.random.default_rng(seed)
    permuted = [ordered[i] for i in rng.permutation(len(ordered))]
    n_val = max(1, round(val_fraction * len(ordered)))
    return set(permuted[n_val:]), set(permuted[:n_val])

# Агрегатор per-sequence RMSD в метку скаффолда — ЕДИНЫЙ для всего проекта.
# Спецификация MotifBench (docs/07) и постановка скрининга требуют min: остов
# проходит дорогой фильтр, если свернулась ХОТЬ ОДНА из восьми последовательностей.
# До 2026-08-13 везде стоял mean — величина, которой нет ни в бенчмарке, ни в
# постановке: при бимодальном распределении RMSD она измеряет долю удачных
# последовательностей, а не дизайнируемость остова. Обоснование замены и цена
# перехода — analysis/label_choice.py.
SCRMSD_AGG = "min"
AGGREGATORS = {"mean": np.mean, "min": np.min}

# Порог дизайнируемости скаффолда, Å (критерий MotifBench)
DESIGNABLE_MAX_SCRMSD = 2.0


def iter_eval_tables(root: str, oracle: str = "ESMFold"):
    """Итератор (motif, sample, DataFrame) по всем оценкам под root.

    Раскладка MotifBench: <root>/<motif>/<sample>/self_consistency/<csv>.
    """
    pattern = os.path.join(root, "**", ORACLE_CSV[oracle])
    for path in sorted(glob.glob(pattern, recursive=True)):
        if "__MACOSX" in path:
            continue
        parts = path.split(os.sep)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        yield parts[-4], parts[-3], df


def per_sequence_rmsd(
    root: str, oracle: str = "ESMFold", min_sequences: int = 1
) -> dict[tuple[str, str], np.ndarray]:
    """(motif, sample) -> вектор per-sequence RMSD, без агрегации.

    min_sequences отсекает скаффолды со слишком малым числом рефолдов: там,
    где метку делят пополам (потолок оракула), нужно хотя бы 4.
    """
    out = {}
    for motif, sample, df in iter_eval_tables(root, oracle):
        if "rmsd" not in df.columns:
            continue
        rmsd = pd.to_numeric(df["rmsd"], errors="coerce").dropna().to_numpy()
        if len(rmsd) >= max(min_sequences, 1):
            out[(motif, sample)] = rmsd
    return out


def scrmsd_by_scaffold(
    root: str, agg: str = SCRMSD_AGG, oracle: str = "ESMFold"
) -> dict[tuple[str, str], float]:
    """(motif, sample) -> scRMSD скаффолда, агрегированный по рефолдам."""
    aggregate = AGGREGATORS[agg]
    return {
        key: float(aggregate(rmsd))
        for key, rmsd in per_sequence_rmsd(root, oracle).items()
    }


def scaffold_table(root: str, agg: str = SCRMSD_AGG) -> pd.DataFrame:
    """Таблица per-scaffold scRMSD/scTM из всех esm_eval_results.csv.

    scTM берётся согласованно с агрегатором scRMSD: при min — у лучшей
    последовательности, а не усредняется по всем.
    """
    rows = []
    for motif, sample, df in iter_eval_tables(root):
        if "rmsd" not in df.columns or "tm_score" not in df.columns:
            continue
        rmsd = pd.to_numeric(df["rmsd"], errors="coerce")
        tm = pd.to_numeric(df["tm_score"], errors="coerce")
        ok = rmsd.notna()
        if not ok.any():
            continue
        rmsd, tm = rmsd[ok], tm[ok]
        if agg == "min":
            best = rmsd.idxmin()
            sc_rmsd, sc_tm = float(rmsd.loc[best]), float(tm.loc[best])
        else:
            sc_rmsd, sc_tm = float(rmsd.mean()), float(tm.mean())
        rows.append(
            {
                "motif": motif,
                "sample": sample,
                "sc_rmsd": sc_rmsd,
                "sc_tm": sc_tm,
                "n_seqs": int(len(rmsd)),
            }
        )
    return pd.DataFrame(rows)


def mpnn_scores(root: str) -> dict[tuple[str, str], dict]:
    """(motif, sample) -> средние ProteinMPNN NLL: по всем остаткам и по дизайну."""
    out = {}
    for motif, sample, df in iter_eval_tables(root):
        if "mpnn_score" not in df.columns or "header" not in df.columns:
            continue
        # header: "T=0.1, sample=1, score=1.0111, global_score=1.3253, ..."
        design = pd.to_numeric(
            df["header"].str.extract(r"score=([\d.]+)")[0], errors="coerce"
        )
        global_score = pd.to_numeric(df["mpnn_score"], errors="coerce")
        if global_score.notna().sum() == 0:
            continue
        out[(motif, sample)] = {
            "mpnn_global": float(global_score.mean()),
            "mpnn_design": float(design.mean()) if design.notna().any() else np.nan,
        }
    return out
