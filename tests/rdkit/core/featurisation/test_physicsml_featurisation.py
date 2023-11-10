from molflux.datasets import featurise_dataset
from molflux.features import load_from_dict


def test_features_atom_num_only(gdb9_dataset):
    atom_num_mapping = {
        1: 0,
        6: 1,
        7: 2,
        8: 3,
    }

    rep_config = {
        "name": "physicsml_features",
        "config": {"atomic_number_mapping": atom_num_mapping, "backend": "rdkit"},
    }

    reps = load_from_dict(rep_config)
    dataset_feated = featurise_dataset(gdb9_dataset, "mol", reps)

    required_cols = [
        "mol::physicsml_coordinates",
        "mol::physicsml_atom_numbers",
        "mol::physicsml_atom_idxs",
    ]

    assert all(x in dataset_feated.column_names for x in required_cols)

    for atom_nums in dataset_feated["mol::physicsml_atom_numbers"]:
        assert isinstance(atom_nums, list)
        for atom_num in atom_nums:
            assert atom_num in atom_num_mapping.values()

    for coords, atom_nums, atom_idxs in zip(
        dataset_feated["mol::physicsml_coordinates"],
        dataset_feated["mol::physicsml_atom_numbers"],
        dataset_feated["mol::physicsml_atom_idxs"],
    ):
        assert len(coords) == len(atom_nums) == len(atom_idxs)


def test_features_atomic_energies(gdb9_dataset):
    atom_num_mapping = {
        1: 0,
        6: 1,
        7: 2,
        8: 3,
    }
    atomic_energies = {
        1: -1,
        6: {0: -2, 1: -3},
        7: {0: -4, -1: -5, 1: -6},
        8: {0: -7, -1: -8},
    }

    rep_config = {
        "name": "physicsml_features",
        "config": {
            "atomic_number_mapping": atom_num_mapping,
            "atomic_energies": atomic_energies,
            "backend": "rdkit",
        },
    }

    reps = load_from_dict(rep_config)
    dataset_feated = featurise_dataset(gdb9_dataset, "mol", reps)

    required_cols = [
        "mol::physicsml_coordinates",
        "mol::physicsml_atom_numbers",
        "mol::physicsml_atom_idxs",
        "mol::physicsml_total_atomic_energy",
    ]

    assert all(x in dataset_feated.column_names for x in required_cols)

    total_atomic_energies = [
        -6,
        -7,
        -9,
        -6,
        -7,
        -11,
        -10,
        -13,
        -10,
        -11,
        -15,
        -16,
        -14,
        -17,
        -17,
        -12,
        -15,
        -19,
        -20,
        -21,
        -18,
        -21,
        -10,
        -13,
        -12,
        -15,
        -16,
        -20,
        -14,
        -14,
        -15,
        -16,
        -17,
        -18,
        -19,
        -20,
        -22,
        -22,
        -18,
        -21,
        -21,
        -24,
        -16,
        -19,
        -19,
        -19,
        -16,
        -19,
        -24,
        -17,
        -18,
        -19,
        -20,
        -22,
        -25,
        -19,
        -20,
        -19,
        -20,
        -24,
        -23,
        -25,
        -18,
        -19,
        -20,
        -21,
        -22,
        -23,
        -26,
        -24,
        -26,
        -23,
        -24,
        -24,
        -25,
        -25,
        -26,
        -27,
        -27,
        -27,
        -25,
        -28,
        -22,
        -25,
        -25,
        -20,
        -23,
        -23,
        -23,
        -21,
        -22,
        -24,
        -22,
        -24,
        -20,
        -23,
        -23,
        -23,
        -26,
        -20,
    ]
    assert dataset_feated["mol::physicsml_total_atomic_energy"] == total_atomic_energies


def test_features(gdb9_dataset):
    atom_num_mapping = {
        1: 0,
        6: 1,
        7: 2,
        8: 3,
    }

    rep_config = {
        "name": "physicsml_features",
        "config": {
            "atomic_number_mapping": atom_num_mapping,
            "atom_features": ["atomic_num", "degree", "formal_charge"],
            "bond_features": ["order", "is_in_ring", "stereo"],
            "backend": "rdkit",
        },
    }

    reps = load_from_dict(rep_config)
    dataset_feated = featurise_dataset(gdb9_dataset, "mol", reps)

    required_cols = [
        "mol::physicsml_coordinates",
        "mol::physicsml_atom_numbers",
        "mol::physicsml_atom_idxs",
        "mol::physicsml_atom_features",
        "mol::physicsml_bond_features",
        "mol::physicsml_bond_idxs",
    ]

    assert all(x in dataset_feated.column_names for x in required_cols)

    for coords, atom_idxs, atom_feats in zip(
        dataset_feated["mol::physicsml_coordinates"],
        dataset_feated["mol::physicsml_atom_idxs"],
        dataset_feated["mol::physicsml_atom_features"],
    ):
        assert len(coords) == len(atom_idxs) == len(atom_feats)
        for atom_feat in atom_feats:
            assert len(atom_feat) == 142

    for bond_idxs, bond_feats in zip(
        dataset_feated["mol::physicsml_bond_idxs"],
        dataset_feated["mol::physicsml_bond_features"],
    ):
        assert len(bond_idxs) == len(bond_idxs)
        for bond_feat in bond_feats:
            assert len(bond_feat) == 12
