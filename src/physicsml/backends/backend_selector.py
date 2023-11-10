from typing import Any, Callable, Dict, Literal

BackendT = Literal["openeye", "rdkit"]


def atom_feature_callables(
    backend: BackendT,
) -> Dict[str, Any]:
    if backend == "openeye":
        from physicsml.backends.openeye_backend import _OPENEYE_ATOM_FEATURES

        return _OPENEYE_ATOM_FEATURES
    elif backend == "rdkit":
        from physicsml.backends.rdkit_backend import _RDKIT_ATOM_FEATURES

        return _RDKIT_ATOM_FEATURES
    else:
        raise KeyError(f"Unknown backend '{backend}'. Try 'openeye' or 'rdkit'.")


def bond_feature_callables(
    backend: BackendT,
) -> Dict[str, Any]:
    if backend == "openeye":
        from physicsml.backends.openeye_backend import _OPENEYE_BOND_FEATURES

        return _OPENEYE_BOND_FEATURES
    elif backend == "rdkit":
        from physicsml.backends.rdkit_backend import _RDKIT_BOND_FEATURES

        return _RDKIT_BOND_FEATURES
    else:
        raise KeyError(f"Unknown backend '{backend}'. Try 'openeye' or 'rdkit'.")


def atoms_or_file_to_bytes(backend: BackendT) -> Callable:
    if backend == "openeye":
        from physicsml.backends.openeye_backend import atoms_or_file_to_oe_mol_bytes

        return atoms_or_file_to_oe_mol_bytes
    elif backend == "rdkit":
        from physicsml.backends.rdkit_backend import (
            atoms_or_file_to_rdkit_mol_bytes,
        )

        return atoms_or_file_to_rdkit_mol_bytes
    else:
        raise KeyError(f"Unknown backend '{backend}'. Try 'openeye' or 'rdkit'.")


def to_mol(backend: BackendT) -> Callable:
    if backend == "openeye":
        from physicsml.backends.openeye_backend import to_oe_mol

        return to_oe_mol
    elif backend == "rdkit":
        from physicsml.backends.rdkit_backend import to_rdkit_mol

        return to_rdkit_mol
    else:
        raise KeyError(f"Unknown backend '{backend}'. Try 'openeye' or 'rdkit'.")


def to_mol_bytes(backend: BackendT) -> Callable:
    if backend == "openeye":
        from physicsml.backends.openeye_backend import to_oe_mol_bytes

        return to_oe_mol_bytes
    elif backend == "rdkit":
        from physicsml.backends.rdkit_backend import to_rdkit_mol_bytes

        return to_rdkit_mol_bytes
    else:
        raise KeyError(f"Unknown backend '{backend}'. Try 'openeye' or 'rdkit'.")
