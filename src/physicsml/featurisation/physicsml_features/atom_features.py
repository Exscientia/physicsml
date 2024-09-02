import logging
from typing import Any

from physicsml.backends.backend_selector import BackendT, atom_feature_callables

logger = logging.getLogger(__name__)


class DuplicateFilter:
    def __init__(self) -> None:
        self.msgs = set()  # type: ignore

    def filter(self, record):  # type: ignore
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv


dup_filter = DuplicateFilter()
logger.addFilter(dup_filter)


def compute_atom_numbers_and_coordinates(
    mol: Any,
    atomic_number_mapping: dict[int, int] | None,
    backend: BackendT,
) -> dict[str, list]:
    mol_atom_numbers = []
    mol_idxs = []

    for atom in mol.GetAtoms():
        mol_idxs.append(atom.GetIdx())

        atom_number = atom_feature_callables(backend)["atomic_num"]["method"](atom)

        if atomic_number_mapping is not None:
            values_dict: dict = atomic_number_mapping
        else:
            values_dict = atom_feature_callables(backend)["atomic_num"]["values"]

        mol_atom_numbers.append(values_dict[atom_number])

    if (len(mol_atom_numbers) > 0) and (len(mol_idxs) > 0):
        mol_idxs, mol_atom_numbers = zip(  # type: ignore[assignment]
            *sorted(zip(mol_idxs, mol_atom_numbers, strict=False)),
            strict=False,
        )

    coords_dict = atom_feature_callables(backend)["coordinates"]["method"](mol)

    mol_coords = [list(coords_dict[idx]) for idx in mol_idxs]

    return {
        "physicsml_atom_numbers": list(mol_atom_numbers),
        "physicsml_coordinates": mol_coords,
        "physicsml_atom_idxs": list(mol_idxs),
    }


def compute_total_atomic_energies(
    mol: Any,
    atomic_energies: dict,
    backend: BackendT,
) -> dict[str, float]:
    total_energy = 0
    for atom in mol.GetAtoms():
        atom_number = atom_feature_callables(backend)["atomic_num"]["method"](atom)
        formal_charge = atom_feature_callables(backend)["formal_charge"]["method"](atom)

        if isinstance(atomic_energies[atom_number], float) or isinstance(
            atomic_energies[atom_number],
            int,
        ):
            atomic_energy = atomic_energies[atom_number]
            if formal_charge != 0:
                logger.warning(
                    f"Atom with atomic number {atom_number} has non-zero charge {formal_charge} but no corresponding atomic energy dict for each charge. Using the only specified atomic energy {atomic_energies[atom_number]}.",
                )
        elif isinstance(atomic_energies[atom_number], dict):
            atomic_energy = atomic_energies[atom_number].get(formal_charge, None)
            if atomic_energy is None:
                raise RuntimeError(
                    f"Atom with atomic number {atom_number} has non-zero charge {formal_charge} but the atomic energy dict provided only has the following charges {list(atomic_energies[atom_number].keys())}.",
                )
        else:
            raise RuntimeError("Malformed atomic_energy dict.")

        total_energy += atomic_energy

    return {"physicsml_total_atomic_energy": total_energy}


def compute_atom_features(
    mol: Any,
    atom_features_list: list[str],
    backend: BackendT,
    one_hot_encoded: bool,
) -> dict[str, list]:
    mol_features = []
    mol_idxs = []

    for atom in mol.GetAtoms():
        mol_idxs.append(atom.GetIdx())
        atom_features = []

        for feature_name in atom_features_list:
            feat = atom_feature_callables(backend)[feature_name]["method"](atom)

            values_dict = atom_feature_callables(backend)[feature_name]["values"]
            if values_dict is not None:
                feat_value: int = values_dict[feat]
            else:
                feat_value = feat

            if one_hot_encoded and isinstance(feat_value, int):
                one_hot_feat: list = [0] * len(values_dict)
                one_hot_feat[feat_value] = 1
                atom_features += one_hot_feat
            else:
                atom_features += [feat_value]

        mol_features.append(atom_features)

    if (len(mol_features) > 0) and (len(mol_idxs) > 0):
        mol_idxs, mol_features = zip(  # type: ignore[assignment]
            *sorted(zip(mol_idxs, mol_features, strict=False)),
            strict=False,
        )

    return {
        "physicsml_atom_features": list(mol_features),
    }
