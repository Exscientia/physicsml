from typing import Tuple

import torch

from .geometry import compute_cell_shifts
from .utils import get_cell_shift_idx, get_number_of_cell_repeats, strides_of


def ravel_3d(idx_3d: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Convert 3d indices meant for an array of sizes `shape` into linear
    indices.

    Parameters
    ----------
    idx_3d : [-1, 3]
        _description_
    shape : [3]
        _description_

    Returns
    -------
    torch.Tensor
        linear indices
    """
    idx_linear = idx_3d[:, 2] + shape[2] * (idx_3d[:, 1] + shape[1] * idx_3d[:, 0])
    return idx_linear


def unravel_3d(idx_linear: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Convert linear indices meant for an array of sizes `shape` into 3d indices.

    Parameters
    ----------
    idx_linear : torch.Tensor [-1]

    shape : torch.Tensor [3]


    Returns
    -------
    torch.Tensor [-1, 3]

    """
    idx_3d = idx_linear.new_empty((idx_linear.shape[0], 3))
    idx_3d[:, 2] = torch.remainder(idx_linear, shape[2])
    idx_3d[:, 1] = torch.remainder(
        torch.div(idx_linear, shape[2], rounding_mode="floor"),
        shape[1],
    )
    idx_3d[:, 0] = torch.div(
        idx_linear,
        shape[1] * shape[2],
        rounding_mode="floor",
    )
    return idx_3d


def get_linear_bin_idx(
    cell: torch.Tensor,
    pos: torch.Tensor,
    nbins_s: torch.Tensor,
) -> torch.Tensor:
    """Find the linear bin index of each input pos given a box defined by its cell vectors and a number of bins, contained in the box, for each directions of the box.

    Parameters
    ----------
    cell : torch.Tensor [3, 3]
        cell vectors
    pos : torch.Tensor [-1, 3]
        set of positions
    nbins_s : torch.Tensor [3]
        number of bins in each directions

    Returns
    -------
    torch.Tensor
        linear bin index
    """
    scaled_pos = torch.linalg.solve(cell.t(), pos.t()).t()
    scaled_pos = torch.stack(
        [scaled_pos[:, 0], scaled_pos[:, 1], scaled_pos[:, 2]],
        dim=1,
    )  # redundant op to make torchscript not fail
    bin_index_s = torch.floor(scaled_pos * nbins_s).to(torch.long)
    bin_index_l = ravel_3d(bin_index_s, nbins_s)
    return bin_index_l


def scatter_bin_index(
    nbins: int,
    max_n_atom_per_bin: int,
    n_images: int,
    bin_index: torch.Tensor,
):
    """convert the linear table `bin_index` into the table `bin_id`. Empty entries in `bin_id` are set to `n_images` so that they can be removed later.

    Parameters
    ----------
    nbins : _type_
        total number of bins
    max_n_atom_per_bin : _type_
        maximum number of atoms per bin
    n_images : _type_
        total number of atoms counting the pbc replicas
    bin_index : _type_
        map relating `atom_index` to the `bin_index` that it belongs to such that `bin_index[atom_index] -> bin_index`.

    Returns
    -------
    bin_id : torch.Tensor [nbins, max_n_atom_per_bin]
        relate `bin_index` (row) with the `atom_index` (stored in the columns).
    """
    device = bin_index.device
    sorted_bin_index, sorted_id = torch.sort(bin_index)
    bin_id = torch.full(
        (nbins * max_n_atom_per_bin,),
        n_images,
        device=device,
        dtype=torch.long,
    )
    sorted_bin_id = torch.remainder(
        torch.arange(bin_index.shape[0], device=device),
        max_n_atom_per_bin,
    )
    sorted_bin_id = sorted_bin_index * max_n_atom_per_bin + sorted_bin_id
    bin_id.scatter_(dim=0, index=sorted_bin_id, src=sorted_id)
    bin_id = bin_id.view((nbins, max_n_atom_per_bin))
    return bin_id


def linked_cell(
    pos: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    num_repeats: torch.Tensor,
    self_interaction: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Determine the atomic neighborhood of the atoms of a given structure for a particular cutoff using the linked cell algorithm.

    Parameters
    ----------
    pos : torch.Tensor [n_atom, 3]
        atomic positions in the unit cell (positions outside the cell boundaries will result in an undifined behaviour)
    cell : torch.Tensor [3, 3]
        unit cell vectors in the format V=[v_0, v_1, v_2]
    cutoff : float
        length used to determine neighborhood
    num_repeats : torch.Tensor [3]
        number of unit cell repetitions in each directions required to account for PBC
    self_interaction : bool, optional
        to keep the original atoms as their own neighbor, by default False

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        neigh_atom : [2, n_neighbors]
            indices of the original atoms (neigh_atom[0]) with their neighbor index (neigh_atom[1]). The indices are meant to access the provided position array
        neigh_shift_idx : [n_neighbors, 3]
            cell shift indices to be used in reconstructing the neighbor atom positions.
    """
    device = pos.device
    dtype = pos.dtype
    n_atom = pos.shape[0]
    # find all the integer shifts of the unit cell given the cutoff and periodicity
    shifts_idx = get_cell_shift_idx(num_repeats, dtype)
    n_cell_image = shifts_idx.shape[0]
    shifts_idx = torch.repeat_interleave(
        shifts_idx,
        n_atom,
        dim=0,
        output_size=n_atom * n_cell_image,
    )
    batch_image = torch.zeros((shifts_idx.shape[0]), dtype=torch.long)
    cell_shifts = compute_cell_shifts(
        cell.view(-1, 3, 3),
        shifts_idx,
        batch_image,
    )

    i_ids = torch.arange(n_atom, device=device, dtype=torch.long)
    i_ids = i_ids.repeat(n_cell_image)
    # compute the positions of the replicated unit cell (including the original)
    # they are organized such that: 1st n_atom are the non-shifted atom, 2nd n_atom are moved by the same translation, ...
    images = pos[i_ids] + cell_shifts
    n_images = images.shape[0]
    # create a rectangular box at [0,0,0] that encompasses all the atoms (hence shifting the atoms so that they lie inside the box)
    b_min = images.min(dim=0).values  # noqa
    b_max = images.max(dim=0).values  # noqa
    images -= b_min - 1e-5
    box_length = b_max - b_min + 1e-3
    # divide the box into square bins of size cutoff in 3d
    nbins_s = torch.maximum(torch.ceil(box_length / cutoff), pos.new_ones(3))
    # adapt the box lengths so that it encompasses
    box_vec = torch.diag_embed(nbins_s * cutoff)
    nbins_s = nbins_s.to(torch.long)
    nbins = int(torch.prod(nbins_s))
    # determine which bins the original atoms and the images belong to following a linear indexing of the 3d bins
    bin_index_j = get_linear_bin_idx(box_vec, images, nbins_s)
    n_atom_j_per_bin = torch.bincount(bin_index_j, minlength=nbins)
    max_n_atom_per_bin = int(n_atom_j_per_bin.max())
    # convert the linear map bin_index_j into a 2d map. This allows for
    # fully vectorized neighbor assignment
    bin_id_j = scatter_bin_index(
        nbins,
        max_n_atom_per_bin,
        n_images,
        bin_index_j,
    )

    # find which bins the original atoms belong to
    bin_index_i = bin_index_j[:n_atom]
    i_bins_l = torch.unique(bin_index_i)
    i_bins_s = unravel_3d(i_bins_l, nbins_s)

    # find the bin indices in the neighborhood of i_bins_l. Since the bins have
    # a side length of cutoff only 27 bins are in the neighborhood
    # (including itself)
    dd = torch.tensor([0, 1, -1], dtype=torch.long, device=device)
    bin_shifts = torch.cartesian_prod(dd, dd, dd)
    n_neigh_bins = bin_shifts.shape[0]
    bin_shifts = bin_shifts.repeat((i_bins_s.shape[0], 1))
    neigh_bins_s = (
        torch.repeat_interleave(
            i_bins_s,
            n_neigh_bins,
            dim=0,
            output_size=n_neigh_bins * i_bins_s.shape[0],
        )
        + bin_shifts
    )
    # some of the generated bin_idx might not be valid
    mask = torch.all(
        torch.logical_and(neigh_bins_s < nbins_s.view(1, 3), neigh_bins_s >= 0),
        dim=1,
    )

    # remove the bins that are outside of the search range, i.e. beyond the borders of the box in the case of non-periodic directions.
    neigh_j_bins_l = ravel_3d(neigh_bins_s[mask], nbins_s)

    max_neigh_per_atom = max_n_atom_per_bin * n_neigh_bins
    # the i_bin related to neigh_j_bins_l
    repeats = mask.view(-1, n_neigh_bins).sum(dim=1)
    neigh_i_bins_l = torch.cat(
        [
            torch.arange(rr, device=device) + i_bins_l[ii] * n_neigh_bins
            for ii, rr in enumerate(repeats)
        ],
        dim=0,
    )
    # the linear neighborlist. make it at large as necessary
    neigh_atom = torch.empty(
        (2, n_atom * max_neigh_per_atom),
        dtype=torch.long,
        device=device,
    )
    # fill the i_atom index
    neigh_atom[0] = (
        torch.arange(n_atom).view(-1, 1).repeat(1, max_neigh_per_atom).view(-1)
    )
    # relate `bin_index` (row) with the `neighbor_atom_index` (stored in the columns). empty entries are set to `n_images`
    bin_id_ij = torch.full(
        (nbins * n_neigh_bins, max_n_atom_per_bin),
        n_images,
        dtype=torch.long,
        device=device,
    )
    # fill the bins with neighbor atom indices
    bin_id_ij[neigh_i_bins_l] = bin_id_j[neigh_j_bins_l]
    bin_id_ij = bin_id_ij.view((nbins, max_neigh_per_atom))
    # map the neighbors in the bins to the central atoms
    neigh_atom[1] = bin_id_ij[bin_index_i].view(-1)
    # remove empty entries
    neigh_atom = neigh_atom[:, neigh_atom[1] != n_images]

    if not self_interaction:
        # neighbor atoms are still indexed from 0 to n_atom*n_cell_image
        neigh_atom = neigh_atom[:, neigh_atom[0] != neigh_atom[1]]

    # sort neighbor list so that the i_atom indices increase
    sorted_ids = torch.argsort(neigh_atom[0])
    neigh_atom = neigh_atom[:, sorted_ids]
    # get the cell shift indices for each neighbor atom
    neigh_shift_idx = shifts_idx[neigh_atom[1]]
    # make sure the j_atom indices access the original positions
    neigh_atom[1] = torch.remainder(neigh_atom[1], n_atom)
    # print(neigh_atom)
    return neigh_atom, neigh_shift_idx


def build_linked_cell_neighborhood(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    n_atoms: torch.Tensor,
    self_interaction: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the neighborlist of a given set of atomic structures using the linked cell algorithm.

    Parameters
    ----------
    positions : torch.Tensor [-1, 3]
        set of atomic positions for each structures
    cell : torch.Tensor [3*n_structure, 3]
        set of unit cell vectors for each structures
    pbc : torch.Tensor [n_structures, 3] bool
        periodic boundary conditions to apply
    cutoff : float
        length used to determine neighborhood
    n_atoms : torch.Tensor
        number of atoms in each structures
    self_interaction : bool
        to keep the original atoms as their own neighbor

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        mapping : [2, n_neighbors]
            indices of the neighbor list for the given positions array, mapping[0/1] correspond respectively to the central/neighbor atom (or node in the graph terminology)
        batch_mapping : [n_neighbors]
            indices mapping the neighbor atom to each structures
        cell_shifts_idx : [n_neighbors, 3]
            cell shift indices to be used in reconstructing the neighbor atom positions.
    """

    n_structure = n_atoms.shape[0]
    device = positions.device
    cell = cell.view((-1, 3, 3))
    pbc = pbc.view((-1, 3))
    # compute the number of cell replica necessary so that all the unit cell's atom have a complete neighborhood (no MIC assumed here)
    num_repeats = get_number_of_cell_repeats(cutoff, cell, pbc)

    stride = strides_of(n_atoms)

    mapping, batch_mapping, cell_shifts_idx = [], [], []
    for i_structure in range(n_structure):
        # compute the neighborhood with the linked cell algorithm
        neigh_atom, neigh_shift_idx = linked_cell(
            positions[stride[i_structure] : stride[i_structure + 1]],
            cell[i_structure],
            cutoff,
            num_repeats[i_structure],
            self_interaction,
        )

        batch_mapping.append(
            i_structure
            * torch.ones(neigh_atom.shape[1], dtype=torch.long, device=device),
        )
        # shift the mapping indices so that they can access positions
        mapping.append(neigh_atom + stride[i_structure])
        cell_shifts_idx.append(neigh_shift_idx)
    return (
        torch.cat(mapping, dim=1),
        torch.cat(batch_mapping, dim=0),
        torch.cat(cell_shifts_idx, dim=0),
    )
