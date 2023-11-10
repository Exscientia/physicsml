from typing import Any, List, Optional

try:
    from openeye import oechem
except ImportError as err:
    from physicsml.utils import OptionalDependencyImportError

    raise OptionalDependencyImportError("OpenEye", "openeye-toolkits") from err


def atoms_or_file_to_oe_mol_bytes(
    atom_list: Optional[List[int]] = None,
    coordinates: Optional[List[List[float]]] = None,
    system_path: Optional[str] = None,
) -> Any:
    assert (atom_list is not None) ^ (system_path is not None), ValueError(
        "Must specify either 'atom_list' or 'system_path' (but not both).",
    )

    if atom_list is not None:
        system = oechem.OEMol()
        for atom_num in atom_list:
            system.NewAtom(atom_num)
        if coordinates is not None:
            flat_coordinates = [xs for x in coordinates for xs in x]
            system.SetCoords(flat_coordinates)
            system.SetDimension(3)
    elif system_path is not None:
        system = next(oechem.oemolistream(system_path).GetOEGraphMols())
    else:
        raise ValueError("Unable to load system.")

    system_bytes = oechem.OEWriteMolToBytes(".oeb", system)

    return system_bytes


def oemol_from_bytes(bytes_molecule: bytes) -> oechem.OEMolBase:
    """Returns a OEMol object representing the input bytes string."""
    oemol = oechem.OEMol()
    if oechem.OEReadMolFromBytes(oemol, ".oeb", bytes_molecule):
        return oemol


def to_oe_mol(molecule: Any) -> oechem.OEMolBase:
    """
    Converts a single sample to OEMol object.
    """

    if isinstance(molecule, oechem.OEMolBase):
        return molecule

    elif isinstance(molecule, bytes):
        return oemol_from_bytes(molecule)

    else:
        raise TypeError(f"Unsupported input sample type: {type(molecule)!r}")


def to_oe_mol_bytes(molecule: Any) -> bytes:
    """
    Converts a single sample to OEMol object.
    """

    if isinstance(molecule, bytes):
        return molecule

    elif isinstance(molecule, oechem.OEMolBase):
        molecule_bytes: bytes = oechem.OEWriteMolToBytes(".oeb", molecule)
        return molecule_bytes

    else:
        raise TypeError(f"Unsupported input sample type: {type(molecule)!r}")


def GetAtomStereoTag(atom: oechem.OEAtomBase) -> Any:
    nbhr_atom_vector = oechem.OEAtomVector(list(atom.GetAtoms()))
    return atom.GetStereo(nbhr_atom_vector, oechem.OEAtomStereo_Tetrahedral)


def GetBondStereoTag(bond: oechem.OEBondBase) -> Any:
    bond_atoms = [bond.GetBgn(), bond.GetEnd()]
    bond_vector = oechem.OEAtomVector(bond_atoms)
    return bond.GetStereo(bond_vector, oechem.OEBondStereo_CisTrans)


_OPENEYE_ATOM_FEATURES = {
    "coordinates": {
        "method": lambda x: x.GetCoords(),
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
        "method": lambda x: GetAtomStereoTag(x),
        "values": {
            oechem.OEAtomStereo_Undefined: 0,
            oechem.OEAtomStereo_Right: 1,
            oechem.OEAtomStereo_Left: 2,
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
    "type": {
        "method": lambda x: x.GetType(),
        "values": {"0": 0, "1": 1},
    },
}

_OPENEYE_BOND_FEATURES = {
    "order": {
        "method": lambda x: x.GetOrder(),
        "values": {x: i for i, x in enumerate(range(1, 5))},
    },
    "stereo": {
        "method": lambda x: GetBondStereoTag(x),
        "values": {
            oechem.OEBondStereo_Cis: 0,
            oechem.OEBondStereo_Trans: 1,
            oechem.OEBondStereo_Undefined: 2,
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
    "is_rotatable": {
        "method": lambda x: x.IsRotor(),
        "values": {
            False: 0,
            True: 1,
        },
    },
    "begin_index": {
        "method": lambda x: x.GetBgnIdx(),
    },
    "end_index": {
        "method": lambda x: x.GetEndIdx(),
    },
}
