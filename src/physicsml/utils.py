import tempfile
from pathlib import Path
from typing import Any, Dict, List

import molflux.core as molflux_core
import numpy as np
from colorama import Fore, Style

from physicsml.backends.backend_selector import BackendT, to_mol


class OptionalDependencyImportError(Exception):

    """Raisable on ImportErrors due to missing optional dependencies."""

    def __init__(self, dependency: str, package: str):
        message = (
            f"Optional dependency {dependency} missing."
            + f"{Style.DIM}# have you tried running the following?{Style.RESET_ALL}\n"
            + f"$ {Style.BRIGHT + Fore.GREEN}pip install '{package}'{Style.RESET_ALL}"
        )

        super().__init__(message)


class OptionalCondaDependencyImportError(Exception):

    """Raisable on ImportErrors due to missing optional dependencies."""

    def __init__(self, dependency: str, package: str):
        message = (
            f"Optional dependency {dependency} missing."
            + f"{Style.DIM}# have you tried running the following?{Style.RESET_ALL}\n"
            + f"$ {Style.BRIGHT + Fore.GREEN}conda install -c conda-forge '{package}'{Style.RESET_ALL}"
        )

        super().__init__(message)


def load_from_dvc(repo_url: str, rev: str, model_path_in_repo: str) -> Any:
    try:
        from dvc.api import DVCFileSystem  # pyright: ignore
    except ImportError as err:
        raise OptionalDependencyImportError("DVC", "dvc[s3]") from err
    else:
        fs = DVCFileSystem(repo_url, rev=rev, subrepos=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            fs.download(
                model_path_in_repo,
                str(Path(tmpdir) / "model"),
                recursive=True,
            )
            model = molflux_core.load_model(str(Path(tmpdir) / "model"))
            featurisation_metadata = molflux_core.load_featurisation_metadata(
                str(Path(tmpdir) / "model" / "featurisation_metadata.json"),
            )

        return model, featurisation_metadata


def get_atomic_energies(
    list_mol_bytes: List[bytes],
    energies: List[float],
    backend: BackendT = "openeye",
) -> Dict[str, float]:
    list_of_mol_atoms = []
    for mol_bytes in list_mol_bytes:
        mol = to_mol(backend)(mol_bytes)
        mol_atoms = []
        for atom in mol.GetAtoms():
            mol_atoms.append(atom.GetAtomicNum())
        list_of_mol_atoms.append(mol_atoms)

    unique_atoms = sorted({xs for x in list_of_mol_atoms for xs in x})
    map = {atom: idx for idx, atom in enumerate(unique_atoms)}
    list_of_mol_atoms = [[map[x] for x in mol_atoms] for mol_atoms in list_of_mol_atoms]

    list_atom_counts = [[0] * len(unique_atoms) for _ in range(len(list_mol_bytes))]
    for idx, mol in enumerate(list_of_mol_atoms):
        for atom in mol:
            list_atom_counts[idx][atom] += 1

    least_squares_fit = np.linalg.lstsq(
        np.array(list_atom_counts),
        np.array(energies),
        rcond=None,
    )[0]
    self_energy_dict = {
        unique_atoms[idx]: energy for idx, energy in enumerate(least_squares_fit)
    }
    return self_energy_dict
