"""
common/ranking.py
-----------------
Общая арифметика внутримотивного ранжирования: точность на вершине списка,
lift относительно базовой доли и деление метки пополам для оценки потолка
оракула.

`precision_at_top` был скопирован в трёх файлах, ещё два держали ту же формулу
под именами `lift` и `lift_from_ranking`; цикл split-half — в пяти. Общие
константы (глубина топа, число разбиений, границы насыщения) разъезжались
между копиями.
"""

import numpy as np

TOP_FRAC = 0.10             # глубина топа, на которой меряется точность
N_SPLITS = 50               # число случайных делений метки пополам
MIN_SAMPLES_PER_MOTIF = 10  # меньше — статистика по мотиву не имеет смысла
MIN_SEQUENCES_FOR_SPLIT = 4 # чтобы половина метки была хотя бы из двух рефолдов
RNG_SEED = 42

# Мотивы, где почти всё или почти ничего не дизайнируемо, ранжирование не
# проверяют: там base_rate упирается в 0 или 1 и lift вырождается.
SATURATED_LOW = 0.10
SATURATED_HIGH = 0.90


def precision_at_top(
    rank_by: np.ndarray, design: np.ndarray, top_frac: float = TOP_FRAC
) -> float:
    """Доля дизайнируемых в топ-top_frac списка, отсортированного по rank_by.

    rank_by — «меньше = лучше» (предсказанный scRMSD, метка оракула).
    """
    k = max(1, round(top_frac * len(design)))
    return float(design[np.argsort(rank_by)[:k]].mean())


def lift_at_top(
    rank_by: np.ndarray, design: np.ndarray, top_frac: float = TOP_FRAC
) -> float:
    """precision@top / базовая доля. NaN на насыщенных мотивах (base 0 или 1)."""
    base = design.mean()
    if base == 0 or base == 1:
        return np.nan
    return precision_at_top(rank_by, design, top_frac) / base


def split_half(
    samples: list[np.ndarray], agg, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Две независимые оценки метки: рефолды каждого скаффолда делятся 4 + 4.

    Возвращает (a, b) — по одному значению агрегата на скаффолд с каждой
    половины. Половина A ранжирует, половина B проверяет: так модель и оракул
    оказываются в одинаковых условиях.
    """
    a, b = [], []
    for s in samples:
        idx = rng.permutation(len(s))
        half = len(s) // 2
        a.append(agg(s[idx[:half]]))
        b.append(agg(s[idx[half:]]))
    return np.array(a), np.array(b)
