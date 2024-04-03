import torch

from .geometry import compute_cell_shifts
from .linked_cell import build_linked_cell_neighborhood
from .naive_impl import build_naive_neighborhood


def strict_nl(
    cutoff: float,
    pos: torch.Tensor,
    cell: torch.Tensor,
    mapping: torch.Tensor,
    batch_mapping: torch.Tensor,
    shifts_idx: torch.Tensor,
):
    """Apply a strict cutoff to the neighbor list defined in mapping.

    Parameters
    ----------
    cutoff : _type_
        _description_
    pos : _type_
        _description_
    cell : _type_
        _description_
    mapping : _type_
        _description_
    batch_mapping : _type_
        _description_
    shifts_idx : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    cell_shifts = compute_cell_shifts(cell, shifts_idx, batch_mapping)
    if cell_shifts is None:
        d2 = (pos[mapping[0]] - pos[mapping[1]]).square().sum(dim=1)
    else:
        d2 = (pos[mapping[0]] - pos[mapping[1]] - cell_shifts).square().sum(dim=1)

    mask = d2 < cutoff * cutoff
    mapping = mapping[:, mask]
    mapping_batch = batch_mapping[mask]
    shifts_idx = shifts_idx[mask]
    return mapping, mapping_batch, shifts_idx


@torch.jit.script
def compute_neighborlist_n2(
    cutoff: float,
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor,
    self_interaction: bool = False,
):
    """Compute the neighborlist for a set of atomic structures using the naive a neighbor search before applying a strict `cutoff`. The atoms positions
    `pos` should be wrapped inside their respective unit cells.

    Parameters
    ----------
    cutoff : float
        cutoff radius of used for the neighbor search
    pos : torch.Tensor [n_atom, 3]
        set of atoms positions wrapped inside their respective unit cells
    cell : torch.Tensor [3*n_structure, 3]
        unit cell vectors in the format [a_1, a_2, a_3]
    pbc : torch.Tensor [n_structure, 3] bool
        periodic boundary conditions to apply. Partial PBC are not supported yet
    batch : torch.Tensor torch.long [n_atom,]
        index of the structure in which the atom belongs to
    self_interaction : bool, optional
        to keep the center atoms as their own neighbor, by default False

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        mapping : [2, n_neighbors]
            indices of the neighbor list for the given positions array, mapping[0/1] correspond respectively to the central/neighbor atom (or node in the graph terminology)
        batch_mapping : [n_neighbors]
            indices mapping the neighbor atom to each structures
        shifts_idx : [n_neighbors, 3]
            cell shift indices to be used in reconstructing the neighbor atom positions.
    """
    n_atoms = torch.bincount(batch)
    mapping, batch_mapping, shifts_idx = build_naive_neighborhood(
        pos,
        cell,
        pbc,
        cutoff,
        n_atoms,
        self_interaction,
    )
    mapping, mapping_batch, shifts_idx = strict_nl(
        cutoff,
        pos,
        cell,
        mapping,
        batch_mapping,
        shifts_idx,
    )
    return mapping, mapping_batch, shifts_idx


@torch.jit.script
def compute_neighborlist(
    cutoff: float,
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor,
    self_interaction: bool = False,
):
    """Compute the neighborlist for a set of atomic structures using the linked
    cell algorithm before applying a strict `cutoff`. The atoms positions `pos`
    should be wrapped inside their respective unit cells.

    Parameters
    ----------
    cutoff : float
        cutoff radius of used for the neighbor search
    pos : torch.Tensor [n_atom, 3]
        set of atoms positions wrapped inside their respective unit cells
    cell : torch.Tensor [3*n_structure, 3]
        unit cell vectors in the format [a_1, a_2, a_3]
    pbc : torch.Tensor [n_structure, 3] bool
        periodic boundary conditions to apply. Partial PBC are not supported yet
    batch : torch.Tensor torch.long [n_atom,]
        index of the structure in which the atom belongs to
    self_interaction : bool, optional
        to keep the center atoms as their own neighbor, by default False

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        mapping : [2, n_neighbors]
            indices of the neighbor list for the given positions array, mapping[0/1] correspond respectively to the central/neighbor atom (or node in the graph terminology)
        batch_mapping : [n_neighbors]
            indices mapping the neighbor atom to each structures
        shifts_idx : [n_neighbors, 3]
            cell shift indices to be used in reconstructing the neighbor atom positions.
    """
    n_atoms = torch.bincount(batch)
    mapping, batch_mapping, shifts_idx = build_linked_cell_neighborhood(
        pos,
        cell,
        pbc,
        cutoff,
        n_atoms,
        self_interaction,
    )

    mapping, mapping_batch, shifts_idx = strict_nl(
        cutoff,
        pos,
        cell,
        mapping,
        batch_mapping,
        shifts_idx,
    )
    return mapping, mapping_batch, shifts_idx
