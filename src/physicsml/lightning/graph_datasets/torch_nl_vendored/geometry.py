from typing import Optional

import torch


def compute_distances(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    assert mapping.dim() == 2
    assert mapping.shape[0] == 2

    if cell_shifts is None:
        dr = pos[mapping[1]] - pos[mapping[0]]
    else:
        dr = pos[mapping[1]] - pos[mapping[0]] + cell_shifts

    return dr.norm(p=2, dim=1)


def compute_cell_shifts(
    cell: torch.Tensor,
    shifts_idx: torch.Tensor,
    batch_mapping: torch.Tensor,
):
    if cell is None:
        cell_shifts = None
    else:
        cell_shifts = torch.einsum(
            "jn,jnm->jm",
            shifts_idx,
            cell.view(-1, 3, 3)[batch_mapping],
        )
    return cell_shifts
