"""
dataset/sequence_clusters.py
----------------------------
Группировка PDB-записей по кластерам последовательностей RCSB (30% идентичности).

Зачем: сплит по цепи разводит по разным сплитам цепи одной и той же записи —
у гомоолигомера это буквально одна последовательность в почти одной
конформации (замер 14.08.2026: 58.2% test-цепей имели «однофамильца» в train),
а гомологов из разных записей он не ловит вовсе.

RCSB публикует готовую кластеризацию сущностей по идентичности, файл
`clusters-by-entity-30.txt`: одна строка — один кластер, в строке
идентификаторы вида `1ABC_1` (запись_сущность).

Как записи превращаются в группы. Наш ключ — цепь (`pdb_id` + `chain_id`), а
кластеры заданы по сущностям, и соответствия цепь → сущность у нас нет.
Поэтому запись получает идентификатор своего МИНИМАЛЬНОГО кластера. Слияние
записей транзитивно через общие кластеры пробовалось и отброшено: у записей с
несколькими сущностями цепочки склеек схлопывают 90% датасета в одну группу
(замер: максимальная группа 128 320 цепей из 141 726).

Остаточный риск правила: запись-комплекс может получить представителя по
чужой сущности и разойтись со своим гомологом-мономером. Измерено на нашем
наборе — 15.9% цепей делят кластер с другой группой против 92.4% при сплите
по записи; фактическая доля попавших в разные сплиты меньше и печатается
в статистике `dataset/make_split.py`.
"""

import os
from collections import defaultdict

import requests

CLUSTERS_URL = "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt"
CLUSTERS_PATH = "dataset/clusters-by-entity-30.txt"


def download_clusters(path: str = CLUSTERS_PATH, url: str = CLUSTERS_URL) -> str:
    """Скачивает файл кластеров, если его ещё нет."""
    if os.path.exists(path):
        return path
    print(f"Скачивание кластеров RCSB: {url}")
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fp:
        fp.write(response.content)
    print(f"Сохранено: {path} ({os.path.getsize(path) / 1e6:.1f} МБ)")
    return path


def clusters_by_entry(path: str = CLUSTERS_PATH) -> dict[str, set[int]]:
    """PDB ID (верхний регистр) -> номера кластеров его сущностей."""
    out: dict[str, set[int]] = defaultdict(set)
    with open(path, encoding="utf-8") as fp:
        for cluster_id, line in enumerate(fp):
            for token in line.split():
                out[token.split("_", 1)[0].upper()].add(cluster_id)
    return dict(out)


def entry_groups(path: str = CLUSTERS_PATH) -> dict[str, str]:
    """PDB ID (верхний регистр) -> идентификатор группы гомологов."""
    return {
        entry: f"seqclust::{min(clusters)}"
        for entry, clusters in clusters_by_entry(path).items()
    }


def homology_across_groups(entry_to_group: dict[str, str], path: str = CLUSTERS_PATH) -> float:
    """Доля записей, чей кластер встречается больше чем в одной группе.

    Верхняя оценка остаточной гомологии: часть таких записей всё равно окажется
    в одном сплите.
    """
    clusters = clusters_by_entry(path)
    groups_of_cluster: dict[int, set[str]] = defaultdict(set)
    for entry, group in entry_to_group.items():
        for cluster in clusters.get(entry, ()):
            groups_of_cluster[cluster].add(group)
    risky = sum(
        any(len(groups_of_cluster[c]) > 1 for c in clusters.get(entry, ()))
        for entry in entry_to_group
    )
    return risky / max(len(entry_to_group), 1)
