from typing import Any, Dict, List, Optional, Tuple

try:
    from rdkit import Chem
except ImportError as err:
    from physicsml.utils import OptionalDependencyImportError

    raise OptionalDependencyImportError("RDKit", "rdkit") from err


def atoms_or_file_to_rdkit_mol_bytes(
    atom_list: Optional[List[int]] = None,
    coordinates: Optional[List[List[float]]] = None,
    system_path: Optional[str] = None,
) -> Any:
    assert (atom_list is not None) ^ (system_path is not None), ValueError(
        "Must specify either 'atom_list' or 'system_path' (but not both).",
    )

    from rdkit import Chem

    if atom_list is not None:
        system = Chem.RWMol()
        for atom_num in atom_list:
            atom = Chem.Atom(atom_num)
            system.AddAtom(atom)

        conf = Chem.Conformer()
        for i in range(system.GetNumAtoms()):
            if coordinates is not None:
                conf.SetAtomPosition(i, coordinates[i])
            else:
                conf.SetAtomPosition(i, [0.0, 0.0, 0.0])
        system.AddConformer(conf)

    elif system_path is not None:
        system = Chem.MolFromPDBFile(
            system_path,
            sanitize=False,
            removeHs=False,
            proximityBonding=False,
        )
    else:
        raise ValueError("Unable to load system.")

    system_bytes = system.ToBinary()

    return system_bytes


def rdkit_mol_from_bytes(bytes_molecule: bytes) -> Chem.Mol:
    """Returns a RDKit Mol object representing the input bytes string."""
    return Chem.Mol(bytes_molecule)  # S301


def to_rdkit_mol(molecule: Any) -> Chem.Mol:
    """
    Converts a single sample to RDKit object.
    """

    if isinstance(molecule, Chem.Mol):
        return molecule

    elif isinstance(molecule, bytes):
        return rdkit_mol_from_bytes(molecule)

    else:
        raise TypeError(f"Unsupported input sample type: {type(molecule)!r}")


def to_rdkit_mol_bytes(molecule: Any) -> bytes:
    """
    Converts a single sample to OEMol object.
    """

    if isinstance(molecule, bytes):
        return molecule

    elif isinstance(molecule, Chem.Mol):
        return molecule.ToBinary()  # type: ignore

    else:
        raise TypeError(f"Unsupported input sample type: {type(molecule)!r}")


def rdkit_coordinates(mol: Chem.Mol) -> Dict[int, Tuple[float]]:
    coords = mol.GetConformer().GetPositions()
    return dict(enumerate(coords))


_RDKIT_ATOM_FEATURES = {
    "coordinates": {
        "method": lambda x: rdkit_coordinates(x),
    },
    "atomic_num": {
        "method": lambda x: x.GetAtomicNum(),
        "values": {x: i for i, x in enumerate(range(1, 120))},
    },
    "formal_charge": {
        "method": lambda x: x.GetFormalCharge(),
        "values": {x: i for i, x in enumerate(range(-5, 7))},
    },
    "degree": {
        "method": lambda x: x.GetDegree(),
        "values": {x: i for i, x in enumerate(range(11))},
    },
    "stereo": {
        "method": lambda x: x.GetChiralTag(),
        "values": {
            Chem.ChiralType.CHI_UNSPECIFIED: 0,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
        },
    },
    "is_in_ring": {
        "method": lambda x: x.IsInRing(),
        "values": {
            False: 0,
            True: 1,
        },
    },
    "is_aromatic": {
        "method": lambda x: x.IsAromatic(),
        "values": {
            False: 0,
            True: 1,
        },
    },
    "is_chiral": {
        "method": lambda x: x.IsChiral(),
        "values": {
            False: 0,
            True: 1,
        },
    },
}

_RDKIT_BOND_FEATURES = {
    "order": {
        "method": lambda x: x.GetBondType(),
        "values": {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
        },
    },
    "stereo": {
        "method": lambda x: x.GetStereo(),
        "values": {
            Chem.rdchem.BondStereo.STEREOCIS: 0,
            Chem.rdchem.BondStereo.STEREOTRANS: 1,
            Chem.rdchem.BondStereo.STEREOZ: 2,  # Z/E is not directly supported in RDKit
            Chem.rdchem.BondStereo.STEREOE: 3,
            Chem.rdchem.BondStereo.STEREOANY: 4,
            Chem.rdchem.BondStereo.STEREONONE: 5,
        },
    },
    "is_in_ring": {
        "method": lambda x: x.IsInRing(),
        "values": {
            False: 0,
            True: 1,
        },
    },
    "is_chiral": {
        "method": lambda x: x.IsChiral(),
        "values": {
            False: 0,
            True: 1,
        },
    },
    "is_rotatable": {
        "method": lambda x: x.IsRotor(),
        "values": {
            False: 0,
            True: 1,
        },
    },
    "begin_index": {
        "method": lambda x: x.GetBeginAtomIdx(),
    },
    "end_index": {
        "method": lambda x: x.GetEndAtomIdx(),
    },
}
