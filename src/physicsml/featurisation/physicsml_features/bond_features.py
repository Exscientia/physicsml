from typing import Any, Dict, List

from physicsml.backends.backend_selector import BackendT, bond_feature_callables


def compute_bond_features(
    mol: Any,
    bond_features_list: List[str],
    one_hot_encoded: bool,
    backend: BackendT,
) -> Dict[str, List]:
    mol_features = []
    mol_idxs = []

    for bond in mol.GetBonds():
        mol_idxs.append(
            [
                bond_feature_callables(backend)["begin_index"]["method"](bond),
                bond_feature_callables(backend)["end_index"]["method"](bond),
            ],
        )
        bond_features = []

        for feature_name in bond_features_list:
            feat = bond_feature_callables(backend)[feature_name]["method"](bond)

            values_dict = bond_feature_callables(backend)[feature_name]["values"]
            feat_value: int = values_dict[feat]

            if one_hot_encoded:
                one_hot_feat: List = [0] * len(values_dict)
                one_hot_feat[feat_value] = 1
                bond_features += one_hot_feat
            else:
                bond_features += [feat_value]

        mol_features.append(bond_features)

    if (len(mol_features) > 0) and (len(mol_idxs) > 0):
        mol_idxs, mol_features = zip(*sorted(zip(mol_idxs, mol_features)))

    return {
        "physicsml_bond_features": list(mol_features),
        "physicsml_bond_idxs": list(mol_idxs),
    }
