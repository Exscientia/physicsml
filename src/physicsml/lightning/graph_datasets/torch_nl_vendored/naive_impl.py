from typing import Tuple

import torch

from .utils import get_cell_shift_idx, get_number_of_cell_repeats, strides_of


def get_fully_connected_mapping(
    i_ids: torch.Tensor,
    shifts_idx: torch.Tensor,
    self_interaction: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_atom = i_ids.shape[0]
    n_atom2 = n_atom * n_atom
    n_cell_image = shifts_idx.shape[0]
    j_ids = torch.repeat_interleave(
        i_ids,
        n_cell_image,
        dim=0,
        output_size=n_cell_image * n_atom,
    )
    mapping = torch.cartesian_prod(i_ids, j_ids)
    shifts_idx = shifts_idx.repeat((n_atom2, 1))
    if not self_interaction:
        mask = torch.ones(
            mapping.shape[0],
            dtype=torch.bool,
            device=i_ids.device,
        )
        ids = n_cell_image * torch.arange(
            n_atom,
            device=i_ids.device,
        ) + torch.arange(
            0,
            mapping.shape[0],
            n_atom * n_cell_image,
            device=i_ids.device,
        )
        mask[ids] = False
        mapping = mapping[mask, :]
        shifts_idx = shifts_idx[mask]
    return mapping, shifts_idx


def build_naive_neighborhood(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    n_atoms: torch.Tensor,
    self_interaction: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TODO: add doc"""
    device = positions.device
    dtype = positions.dtype

    num_repeats_ = get_number_of_cell_repeats(cutoff, cell, pbc)

    stride = strides_of(n_atoms)
    ids = torch.arange(positions.shape[0], device=device, dtype=torch.long)

    mapping, batch_mapping, shifts_idx_ = [], [], []
    for i_structure in range(n_atoms.shape[0]):
        num_repeats = num_repeats_[i_structure]
        shifts_idx = get_cell_shift_idx(num_repeats, dtype)
        i_ids = ids[stride[i_structure] : stride[i_structure + 1]]

        s_mapping, shifts_idx = get_fully_connected_mapping(
            i_ids,
            shifts_idx,
            self_interaction,
        )
        mapping.append(s_mapping)
        batch_mapping.append(
            torch.full(
                (s_mapping.shape[0],),
                i_structure,
                dtype=torch.long,
                device=device,
            ),
        )
        shifts_idx_.append(shifts_idx)
    return (
        torch.cat(mapping, dim=0).t(),
        torch.cat(batch_mapping, dim=0),
        torch.cat(shifts_idx_, dim=0),
    )
