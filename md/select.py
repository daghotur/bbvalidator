"""
md/select.py
------------
Отбор кандидатов под МД: внутримотивное сравнение топа и низа ранжирования.

Почему внутри мотива, а не по пулу. Пуловый топ-10 `filter_results.csv`
целиком состоит из `17_7DGW` (125 aa), а низ — из `04_5WN9` (75 aa). Сравнение
таких групп измеряло бы разницу мотивов и длин, а не работу скорера; это тот же
артефакт состава, из-за которого docs/08 заменил пуловый top-K на внутримотивную
метрику. Внутри одного мотива длина и сложность задачи скаффолдинга постоянны,
и единственное, чем различаются группы, — предсказание модели.

МД считается не на скаффолде: RFdiffusion выдаёт только backbone с GLY-заглушками
вместо последовательности. Полноатомный объект — ESMFold-рефолд одной из восьми
последовательностей ProteinMPNN. Берётся argmin по RMSD, то есть ровно та
последовательность, которая задаёт метку scRMSD=min (common/motifbench.SCRMSD_AGG).
Это даёт лучший шанс и плохим скаффолдам тоже — иначе контроль был бы нечестным.
"""

import argparse
import os

import pandas as pd

from common import motifbench as mb

FILTER_CSV = "results/filter_results.csv"
GENERATOR = "RFdiffusion"


def best_refold(motif: str, sample: str) -> dict:
    """Путь к ESMFold-рефолду последовательности с минимальным RMSD."""
    base = os.path.join(
        mb.EVAL_SOURCES[GENERATOR], "evaluation", motif, sample, "self_consistency"
    )
    tab = pd.read_csv(os.path.join(base, mb.ORACLE_CSV["ESMFold"]))
    row = tab.loc[tab["rmsd"].idxmin()]
    return {
        "esm_pdb": os.path.join(base, "esmf", f"sample_{int(row['sample_idx'])}.pdb"),
        "seq_idx": int(row["sample_idx"]),
        "seq_rmsd": float(row["rmsd"]),
        "plddt": float(row["plddt"]),
        "ptm": float(row["ptm"]),
        "sequence": str(row["sequence"]),
    }


def scaffold_pdb(motif: str, sample: str) -> str:
    return os.path.join(
        mb.GENERATOR_DIRS[GENERATOR], "scaffolds", motif, f"{sample}.pdb"
    )


def select(motifs: list[str], k: int) -> pd.DataFrame:
    pred = pd.read_csv(FILTER_CSV)
    pred["motif"] = pred["name"].str.replace(r"_\d+\.pdb$", "", regex=True)
    pred["sample"] = pred["name"].str.replace(r"\.pdb$", "", regex=True)

    truth = mb.scaffold_table(mb.EVAL_SOURCES[GENERATOR])
    df = pred.merge(truth, on=["motif", "sample"], how="inner")
    _, val = mb.motif_split(df["motif"].unique())

    rows = []
    for motif in motifs:
        d = df[df.motif == motif].sort_values("pred_scrmsd")
        if d.empty:
            raise SystemExit(f"мотив {motif} не найден в {FILTER_CSV}")
        # Отбор ТОЛЬКО по предсказанию модели: истина не участвует в выборе,
        # иначе тест выродится в «проверим, что дизайнируемое стабильно».
        for grp, part in (("TOP", d.head(k)), ("BOT", d.tail(k))):
            for _, r in part.iterrows():
                ref = best_refold(motif, r["sample"])
                rows.append(
                    {
                        "motif": motif,
                        "sample": r["sample"],
                        "group": grp,
                        "length": int(r["length"]),
                        "role": "val-motif" if motif in val else "in-sample",
                        "pred_scrmsd": float(r["pred_scrmsd"]),
                        "sc_rmsd_true": float(r["sc_rmsd"]),
                        "designable": bool(r["sc_rmsd"] < mb.DESIGNABLE_MAX_SCRMSD),
                        "p_steric": float(r["p_steric"]),
                        "clash_pairs": int(r["clash_pairs"]),
                        "verdict": r["verdict"],
                        "scaffold_pdb": scaffold_pdb(motif, r["sample"]),
                        **ref,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motifs", nargs="+", default=["28_5YUI", "06_6E6R"])
    ap.add_argument("-k", type=int, default=3, help="сколько брать сверху и снизу")
    ap.add_argument("-o", default="md/manifest.csv")
    a = ap.parse_args()

    man = select(a.motifs, a.k)
    missing = [p for p in [*man.esm_pdb, *man.scaffold_pdb] if not os.path.exists(p)]
    if missing:
        raise SystemExit("нет файлов:\n  " + "\n  ".join(missing))

    man.to_csv(a.o, index=False)
    cols = ["motif", "sample", "group", "pred_scrmsd", "sc_rmsd_true",
            "designable", "seq_idx", "plddt"]
    print(man[cols].to_string(index=False))
    print(f"\n{len(man)} систем -> {a.o}")


if __name__ == "__main__":
    main()
