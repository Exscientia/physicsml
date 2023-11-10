from typing import Any, Dict, List, Optional

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.typing import MolArray

from physicsml.backends.backend_selector import BackendT, to_mol
from physicsml.featurisation.physicsml_features.atom_features import (
    compute_atom_features,
    compute_atom_numbers_and_coordinates,
    compute_total_atomic_energies,
)
from physicsml.featurisation.physicsml_features.bond_features import (
    compute_bond_features,
)

_DESCRIPTION = """
Features for physicsml models. Choose from (in the config)

    atom_features = FILL IN
    bond_features = FILL IN

Can also, specify whether one_hot_encoded or not, True by default.
"""


class PhysicsMLFeatures(RepresentationBase):
    def __init__(
        self,
        atomic_number_mapping: Optional[Dict[int, int]] = None,
        atomic_energies: Optional[Dict[int, int]] = None,
        atom_features: Optional[List[str]] = None,
        bond_features: Optional[List[str]] = None,
        one_hot_encoded: bool = True,
        backend: Optional[BackendT] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        assert backend is not None, KeyError("Must specify backend for featurisation")

        # make sure json encoding str keys are converted to int(k)
        self.atomic_number_mapping: Optional[Dict] = {}
        if atomic_number_mapping is not None:
            for k, v in atomic_number_mapping.items():
                self.atomic_number_mapping[int(k)] = v
        else:
            self.atomic_number_mapping = None

        self.atomic_energies: Optional[Dict] = {}
        if atomic_energies is not None:
            for k, v in atomic_energies.items():
                self.atomic_energies[int(k)] = v

            assert self.atomic_number_mapping is not None, RuntimeError(
                "Must specify atomic_number_mapping when specifying atomic_energies",
            )
            for k in self.atomic_number_mapping.keys():
                assert k in self.atomic_energies, RuntimeError(
                    f"Atomic number {int(k)} specified in atomic_number_mapping but not in atomic_energies",
                )
                assert (
                    isinstance(self.atomic_energies[k], float)
                    or isinstance(self.atomic_energies[k], int)
                ) or (
                    isinstance(self.atomic_energies[k], dict)
                    and all(
                        isinstance(kk, int)
                        and (isinstance(vv, float) or isinstance(vv, int))
                        for kk, vv in self.atomic_energies[k].items()
                    )
                ), RuntimeError(
                    f"Atomic energies of atomic number {int(k)} must either be a float (implicitly assumed for neutral atom) or a dictionary of charge: energy. Got {self.atomic_energies[k]}.",
                )
        else:
            self.atomic_energies = None

        if (atom_features is not None) and (len(atom_features) > 0):
            self.atom_features_list: Optional[List[str]] = sorted(atom_features)
        else:
            self.atom_features_list = None

        if (bond_features is not None) and (len(bond_features) > 0):
            self.bond_features_list: Optional[List[str]] = sorted(bond_features)
        else:
            self.bond_features_list = None

        self.one_hot_encoded = one_hot_encoded

        self.backend = backend

    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        mols = (to_mol(self.backend)(sample) for sample in samples)

        list_of_feature_dicts = []

        for mol in mols:
            atom_nums_coords = compute_atom_numbers_and_coordinates(
                mol,
                self.atomic_number_mapping,
                backend=self.backend,
            )

            if self.atomic_energies is not None:
                total_atomic_energy = compute_total_atomic_energies(
                    mol,
                    self.atomic_energies,
                    backend=self.backend,
                )
            else:
                total_atomic_energy = {}

            if self.atom_features_list is not None:
                atom_feat_dict = compute_atom_features(
                    mol,
                    atom_features_list=self.atom_features_list,
                    one_hot_encoded=self.one_hot_encoded,
                    backend=self.backend,
                )
            else:
                atom_feat_dict = {}

            if self.bond_features_list is not None:
                bond_feat_dict = compute_bond_features(
                    mol,
                    bond_features_list=self.bond_features_list,
                    one_hot_encoded=self.one_hot_encoded,
                    backend=self.backend,
                )
            else:
                bond_feat_dict = {}

            list_of_feature_dicts.append(
                {
                    **atom_nums_coords,
                    **total_atomic_energy,
                    **atom_feat_dict,
                    **bond_feat_dict,
                },
            )

        dict_of_features = {
            k: [dic[k] for dic in list_of_feature_dicts]
            for k in list_of_feature_dicts[0]
        }

        sorted_dict = dict(sorted(dict_of_features.items()))

        return sorted_dict
