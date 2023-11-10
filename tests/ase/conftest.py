# type: ignore

import datasets
import h5py
import molflux.core as molflux_core
import numpy as np
import pandas as pd
import pytest

from rdkit import Chem


@pytest.fixture()
def gdb9_dataset():
    mols_iterator = Chem.SDMolSupplier("tests/data/gdb9.sdf", removeHs=False)

    df_gdb9 = pd.read_csv("tests/data/gdb9.csv")

    dataset = []
    for mol, (_index, labels) in zip(mols_iterator, df_gdb9.iterrows()):
        example_dict = labels.to_dict()
        example_dict["mol"] = mol.ToBindary()
        dataset.append(example_dict)

    dataset = datasets.Dataset.from_list(dataset)

    return dataset


@pytest.fixture()
def ani1x_dataset():
    h5_file = h5py.File("tests/data/ani1x.h5", "r+")

    # get columns to loop over
    columns = [
        "mol_bytes",
        "chemical_formula",
        "ccsd(t)_cbs.energy",
        "hf_dz.energy",
        "hf_qz.energy",
        "hf_tz.energy",
        "mp2_dz.corr_energy",
        "mp2_qz.corr_energy",
        "mp2_tz.corr_energy",
        "npno_ccsd(t)_dz.corr_energy",
        "npno_ccsd(t)_tz.corr_energy",
        "tpno_ccsd(t)_dz.corr_energy",
        "wb97x_dz.cm5_charges",
        "wb97x_dz.dipole",
        "wb97x_dz.energy",
        "wb97x_dz.forces",
        "wb97x_dz.hirshfeld_charges",
        "wb97x_dz.quadrupole",
        "wb97x_tz.dipole",
        "wb97x_tz.energy",
        "wb97x_tz.forces",
        "wb97x_tz.mbis_charges",
        "wb97x_tz.mbis_dipoles",
        "wb97x_tz.mbis_octupoles",
        "wb97x_tz.mbis_quadrupoles",
        "wb97x_tz.mbis_volumes",
    ]
    columns.remove("mol_bytes")
    columns.append("coordinates")

    # reset index
    index = 0
    dataset = []

    # loop over chemical formulae (each have multiple conformers)
    for chemical_formula in h5_file:
        # the dict of data for this chemical formula
        chemical_formula_dict = {
            key: value[:].tolist()
            for key, value in dict(h5_file[chemical_formula]).items()
        }

        # atomic numbers list for the chemical formula
        atomic_nums = chemical_formula_dict["atomic_numbers"]
        del chemical_formula_dict["atomic_numbers"]

        # make all the molecules into mol bytes (only atomic numbers and coords available)
        mol_list = []
        for coords in chemical_formula_dict["coordinates"]:
            mol = Chem.RWMol()
            for atomic_num in atomic_nums:
                atom = Chem.Atom(atomic_num)
                mol.AddAtom(atom)
            conf = Chem.Conformer()
            for idx, coord in enumerate(coords):
                conf.SetAtomPosition(idx, coord)
            mol.AddConformer(conf)

            mol_bytes = mol.ToBinary()
            mol_list.append(mol_bytes)

        # delete coordinates from the dict and add mols
        del chemical_formula_dict["coordinates"]
        chemical_formula_dict["mol_bytes"] = mol_list

        # find list of example dicts (swap dict of lists to list of dicts)
        chemical_formula_examples_list = [
            dict(zip(chemical_formula_dict, t))
            for t in zip(*chemical_formula_dict.values())
        ]

        # yield the examples for the conformers
        for example_dict in chemical_formula_examples_list:
            index += 1

            # add chemical formula
            example_dict["chemical_formula"] = chemical_formula
            dataset.append(example_dict)

    dataset = datasets.Dataset.from_list(dataset)

    dataset = dataset.filter(
        lambda x: not np.isnan(np.array(x["wb97x_dz.energy"])).any(),
    )
    dataset = dataset.filter(
        lambda x: not np.isnan(np.array(x["wb97x_dz.hirshfeld_charges"])).any(),
    )
    dataset = dataset.filter(
        lambda x: not np.isnan(np.array(x["wb97x_dz.forces"])).any(),
    )
    dataset = dataset.filter(
        lambda x: not np.isnan(np.array(x["wb97x_dz.dipole"])).any(),
    )

    return dataset


@pytest.fixture()
def featurised_gdb9_atomic_nums(gdb9_dataset):
    # specify config
    featurisation_metadata = {
        "version": 1,
        "config": [
            {
                "column": "mol",
                "representations": [
                    {
                        "name": "physicsml_features",
                        "config": {
                            "atomic_number_mapping": {
                                1: 0,
                                6: 1,
                                7: 2,
                                8: 3,
                            },
                            "atomic_energies": {
                                1: -0.6035075,
                                6: -38.073092,
                                7: -54.753016,
                                8: -75.221873,
                            },
                            "backend": "rdkit",
                        },
                        "as": "{feature_name}",
                    },
                ],
            },
        ],
    }

    # featurise the mols
    dataset_feated = molflux_core.featurise_dataset(
        gdb9_dataset,
        featurisation_metadata=featurisation_metadata,
        num_proc=4,
        batch_size=100,
    )

    x_features = list(set(dataset_feated.column_names) - set(gdb9_dataset.column_names))

    return dataset_feated, x_features, featurisation_metadata


@pytest.fixture()
def featurised_gdb9_atomic_nums_and_feats(gdb9_dataset):
    # specify config
    featurisation_metadata = {
        "version": 1,
        "config": [
            {
                "column": "mol",
                "representations": [
                    {
                        "name": "physicsml_features",
                        "config": {
                            "atomic_number_mapping": {
                                1: 0,
                                6: 1,
                                7: 2,
                                8: 3,
                            },
                            "atomic_energies": {
                                1: -0.6035075,
                                6: -38.073092,
                                7: -54.753016,
                                8: -75.221873,
                            },
                            "atom_features": ["degree", "formal_charge"],
                            "backend": "rdkit",
                        },
                        "as": "{feature_name}",
                    },
                ],
            },
        ],
    }

    # featurise the mols
    dataset_feated = molflux_core.featurise_dataset(
        gdb9_dataset,
        featurisation_metadata=featurisation_metadata,
        num_proc=4,
        batch_size=100,
    )

    x_features = list(set(dataset_feated.column_names) - set(gdb9_dataset.column_names))

    return dataset_feated, x_features, featurisation_metadata


@pytest.fixture()
def featurised_gdb9_atomic_nums_and_feats_and_bond_feats(gdb9_dataset):
    # specify config
    featurisation_metadata = {
        "version": 1,
        "config": [
            {
                "column": "mol",
                "representations": [
                    {
                        "name": "physicsml_features",
                        "config": {
                            "atomic_number_mapping": {
                                1: 0,
                                6: 1,
                                7: 2,
                                8: 3,
                            },
                            "atomic_energies": {
                                1: -0.6035075,
                                6: -38.073092,
                                7: -54.753016,
                                8: -75.221873,
                            },
                            "atom_features": ["degree", "formal_charge"],
                            "bond_features": [
                                "order",
                                "is_in_ring",
                                "stereo",
                            ],
                            "backend": "rdkit",
                        },
                        "as": "{feature_name}",
                    },
                ],
            },
        ],
    }

    # featurise the mols
    dataset_feated = molflux_core.featurise_dataset(
        gdb9_dataset,
        featurisation_metadata=featurisation_metadata,
        num_proc=4,
        batch_size=100,
    )

    x_features = list(set(dataset_feated.column_names) - set(gdb9_dataset.column_names))

    return dataset_feated, x_features, featurisation_metadata


@pytest.fixture()
def featurised_ani1x_atomic_nums(ani1x_dataset):
    # specify config
    featurisation_metadata = {
        "version": 1,
        "config": [
            {
                "column": "mol_bytes",
                "representations": [
                    {
                        "name": "physicsml_features",
                        "config": {
                            "atomic_number_mapping": {
                                1: 0,
                                6: 1,
                                7: 2,
                                8: 3,
                            },
                            "atomic_energies": {
                                1: -0.5894385,
                                6: -38.103158,
                                7: -54.724035,
                                8: -75.196441,
                            },
                            "backend": "rdkit",
                        },
                        "as": "{feature_name}",
                    },
                ],
            },
        ],
    }

    # featurise the mols
    dataset_feated = molflux_core.featurise_dataset(
        ani1x_dataset,
        featurisation_metadata=featurisation_metadata,
        num_proc=4,
        batch_size=100,
    )

    x_features = list(
        set(dataset_feated.column_names) - set(ani1x_dataset.column_names),
    )

    return dataset_feated, x_features, featurisation_metadata
