import h5py
import io
import logging
import concurrent.futures
from typing import List, Optional, Tuple

import numpy as np
import requests
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb
from tqdm import tqdm

# steric_target и hbond_target считаются отдельным скриптом
# dataset/compute_targets.py после завершения параллельной сборки —
# чтобы torch-операции не мешали распараллеливанию воркеров.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_build.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

H5_OUTPUT_FILE = "positive_proteins.h5"
MAX_INSTANCES = 200_000
MIN_LEN, MAX_LEN = 50, 700
PEPTIDE_BOND_MAX_DIST = 2.0


def get_instances(limit: int) -> List[str]:
    logger.info("Формируем запрос к RCSB API")
    q_xray = rcsb.FieldQuery("exptl.method", exact_match="X-RAY DIFFRACTION")
    q_res = rcsb.FieldQuery("rcsb_entry_info.resolution_combined", less_or_equal=2.0)
    q_len = rcsb.FieldQuery("entity_poly.rcsb_sample_sequence_length", range=[MIN_LEN, MAX_LEN])
    q_prot = rcsb.FieldQuery("entity_poly.rcsb_entity_polymer_type", exact_match="Protein")
    comp_query = rcsb.CompositeQuery([q_xray, q_res, q_len, q_prot], "and")

    try:
        instances = list(rcsb.search(comp_query, return_type="polymer_instance"))[
            :limit
        ]
        logger.info(f"Найдено цепей: {len(instances)}")
        return instances
    except Exception as e:
        logger.error(f"Ошибка API запроса: {e}")
        return []


def extract_backbone(chain_struc: struc.AtomArray) -> Optional[np.ndarray]:
    try:
        residue_starts = struc.get_residue_starts(chain_struc)
        coords_list = []

        for i, start in enumerate(residue_starts):
            end = (
                residue_starts[i + 1]
                if i + 1 < len(residue_starts)
                else len(chain_struc)
            )
            res_atoms = chain_struc[start:end]
            names = res_atoms.atom_name

            n_idx = np.where(names == "N")[0]
            ca_idx = np.where(names == "CA")[0]
            c_idx = np.where(names == "C")[0]

            if len(n_idx) == 1 and len(ca_idx) == 1 and len(c_idx) == 1:
                coords_list.append(
                    [
                        res_atoms.coord[n_idx[0]],
                        res_atoms.coord[ca_idx[0]],
                        res_atoms.coord[c_idx[0]],
                    ]
                )

        return np.array(coords_list) if coords_list else None
    except Exception as e:
        logger.debug(f"Ошибка при извлечении backbone: {e}")
        return None


def process_pdb_file(
    pdb_id: str,
    asym_ids: List[str],
) -> List[Tuple[str, Optional[np.ndarray], str, Optional[dict]]]:
    results: List[Tuple[str, Optional[np.ndarray], str, Optional[dict]]] = []
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        cif_file = pdbx.CIFFile.read(io.StringIO(response.text))
        structure = pdbx.get_structure(
            cif_file,
            model=1,
            extra_fields=["label_asym_id", "auth_asym_id", "label_alt_id"],
        )

        for asym_id in asym_ids:
            instance_name = f"{pdb_id}.{asym_id}"
            try:
                if np.any(structure.label_asym_id == asym_id):
                    chain_mask = structure.label_asym_id == asym_id
                elif np.any(structure.auth_asym_id == asym_id):
                    chain_mask = structure.auth_asym_id == asym_id
                elif np.any(structure.chain_id == asym_id):
                    chain_mask = structure.chain_id == asym_id
                else:
                    results.append((instance_name, None, f"Chain {asym_id} not found", None))
                    continue

                chain_mask = chain_mask & struc.filter_amino_acids(structure)
                chain_struc = structure[chain_mask]

                if len(chain_struc) == 0:
                    results.append((instance_name, None, "No standard amino acids", None))
                    continue

                if hasattr(chain_struc, "label_alt_id"):
                    altloc_mask = (chain_struc.label_alt_id == ".") | (
                        chain_struc.label_alt_id == "A"
                    )
                    chain_struc = chain_struc[altloc_mask]

                coords_tensor = extract_backbone(chain_struc)
                if coords_tensor is None:
                    results.append((instance_name, None, "Backbone extraction failed", None))
                    continue

                length = len(coords_tensor)
                if not (MIN_LEN <= length <= MAX_LEN):
                    results.append((instance_name, None, f"Length out of bounds ({length})", None))
                    continue

                c_prev = coords_tensor[:-1, 2, :]
                n_next = coords_tensor[1:, 0, :]
                if np.any(np.linalg.norm(c_prev - n_next, axis=-1) > PEPTIDE_BOND_MAX_DIST):
                    results.append((instance_name, None, "Chain broken", None))
                    continue

                # steric_target и hbond_target проставит compute_targets.py
                targets = {
                    "rmsd_target": 0.0,
                    "steric_target": float("nan"),
                    "hbond_target": float("nan"),
                    "failure_mode_label": 0,
                }

                results.append((instance_name, coords_tensor, "OK", targets))

            except Exception as e:
                results.append((instance_name, None, f"Chain error: {e}", None))

    except Exception as e:
        for asym_id in asym_ids:
            results.append((f"{pdb_id}.{asym_id}", None, f"PDB error: {e}", None))

    return results


def build_dataset(max_workers: int = 4, max_instances: int = MAX_INSTANCES) -> None:
    instances = get_instances(limit=max_instances)
    if not instances:
        logger.warning("Не найдено ни одной цепи. Выход.")
        return

    pdb_groups: dict[str, list[str]] = {}
    for inst in instances:
        pdb_id, asym_id = inst.split(".")
        pdb_groups.setdefault(pdb_id, []).append(asym_id)

    success_count = 0

    with h5py.File(H5_OUTPUT_FILE, "w") as h5f:
        logger.info(f"Запуск параллельной обработки (workers={max_workers})...")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = {
                executor.submit(process_pdb_file, pid, aids): pid
                for pid, aids in pdb_groups.items()
            }

            with tqdm(total=len(instances)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_results = future.result()
                    except Exception as e:
                        logger.error(f"Future failed for {futures[future]}: {e}")
                        continue

                    for instance, coords, status, targets in batch_results:
                        if coords is not None:
                            pdb_id, asym_id = instance.split(".")
                            grp = h5f.create_group(instance)
                            grp.create_dataset(
                                "coords",
                                data=coords.astype(np.float32),
                                compression="gzip",
                            )
                            grp.create_dataset(
                                "label", data=np.array([1.0], dtype=np.float32)
                            )
                            grp.attrs["pdb_id"] = pdb_id
                            grp.attrs["chain_id"] = asym_id
                            grp.attrs["length"] = len(coords)
                            grp.attrs["method"] = "X-RAY"
                            if targets is not None:
                                for k, v in targets.items():
                                    grp.attrs[k] = v
                            success_count += 1
                        pbar.update(1)

    logger.info(f"Собрано цепей: {success_count} / {len(instances)}")


if __name__ == "__main__":
    build_dataset(80, MAX_INSTANCES)
