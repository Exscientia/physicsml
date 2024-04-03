from typing import Optional, Tuple

import torch
from torch_geometric.utils import (
    coalesce,
    to_undirected,
)

from physicsml.lightning.graph_datasets.torch_nl_vendored.neighbor_list import (
    compute_neighborlist,
)

# tracing torch geom funcs to ensure that they use the required overloading
to_undirected_ts = torch.jit.trace(
    to_undirected,
    (
        torch.tensor([[0, 0, 1, 2], [2, 2, 1, 0]]),
        torch.tensor([[0, 1], [0, 1], [1, 2], [2, 1]]).float(),
        torch.tensor(3),
    ),
)
coalesce_ts = torch.jit.trace(
    coalesce,
    (
        torch.tensor([[0, 0, 1, 2], [2, 2, 1, 0]]),
        [
            torch.tensor([[0, 1], [0, 1], [1, 2], [2, 1]]).float(),
            torch.tensor([[0, 1], [0, 1], [1, 2], [2, 1]]).float(),
        ],
        torch.tensor(3),
    ),
)


def compute_neighborlist_n2_no_cell(
    cutoff: float,
    pos: torch.Tensor,
    self_interaction: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r_ij = ((pos.unsqueeze(0) - pos.unsqueeze(1)) ** 2).sum(-1)
    mask = (r_ij <= cutoff**2).nonzero()

    if not self_interaction:
        sender = mask[:, 0]
        receiver = mask[:, 1]
        mask = mask[(sender != receiver)]

    return mask.transpose(0, 1), torch.zeros(
        (mask.shape[0], 3),
        dtype=pos.dtype,
        device=pos.device,
    )


@torch.jit.script
def construct_edge_indices_and_attrs(
    positions: torch.Tensor,
    cutoff: float,
    initial_edge_indices: Optional[torch.Tensor],
    initial_edge_attrs: Optional[torch.Tensor],
    pbc: Optional[Tuple[bool, bool, bool]],
    cell: Optional[torch.Tensor],
    self_interaction: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if (pbc is not None) and (cell is not None):
        # find edge indices
        nbhd_edge_indices, _, nbhd_cell_shift_vector = compute_neighborlist(
            cutoff=cutoff,
            pos=positions,
            cell=cell.type(positions.dtype).to(positions.device),
            pbc=torch.tensor(pbc).to(positions.device),
            batch=torch.zeros(
                positions.shape[0],
                dtype=torch.long,
                device=positions.device,
            ),
            self_interaction=self_interaction,
        )
    else:
        nbhd_edge_indices, nbhd_cell_shift_vector = compute_neighborlist_n2_no_cell(
            cutoff=cutoff,
            pos=positions,
            self_interaction=self_interaction,
        )

    nbhd_edge_indices = nbhd_edge_indices.transpose(0, 1)

    if (
        (initial_edge_indices is not None)
        and (initial_edge_attrs is not None)
        and (len(initial_edge_attrs.shape) > 1)
    ):
        nbhd_edge_indices = nbhd_edge_indices.T
        initial_edge_indices = initial_edge_indices.T

        # make initial edge indices and attrs directional
        initial_edge_indices, initial_edge_attrs = to_undirected_ts(
            initial_edge_indices,
            initial_edge_attrs,
            torch.tensor(positions.shape[0]),
        )
        assert initial_edge_indices is not None
        assert initial_edge_attrs is not None

        # zeros for nbhd edge attrs and initial edges cell shift.
        if len(initial_edge_attrs.shape) == 2:
            zero_nbhd_edge_attrs = torch.zeros(
                nbhd_edge_indices.shape[1],
                initial_edge_attrs.shape[1],
                device=initial_edge_attrs.device,
                dtype=initial_edge_attrs.dtype,
            )
        else:  # attrs shape = (0) for node with no edges
            zero_nbhd_edge_attrs = torch.zeros(
                nbhd_edge_indices.shape[1],
                device=initial_edge_attrs.device,
                dtype=initial_edge_attrs.dtype,
            )
        zero_initial_cell_shift = torch.zeros(
            initial_edge_indices.shape[1],
            3,
            device=nbhd_cell_shift_vector.device,
            dtype=nbhd_cell_shift_vector.dtype,
        )

        # concat. There will be duplicates. Will be removed in next step.
        all_edge_indices = torch.cat([nbhd_edge_indices, initial_edge_indices], dim=1)
        all_edge_attrs = torch.cat([zero_nbhd_edge_attrs, initial_edge_attrs], dim=0)
        all_cell_shit_vector = torch.cat(
            [nbhd_cell_shift_vector, zero_initial_cell_shift],
            dim=0,
        )

        # coalesce to remove duplicates. For attrs, we combine with add. Because we used zeros for the extra edge attrs,
        # this will ensure that we have the correct edge attrs for all edges. Also, because initial edges are contained in
        # nbhd edges, this will also ensure that all edges have correct cell shifts.
        edge_indices, edge_attrs_cell_shift_vector = coalesce_ts(
            all_edge_indices,
            [all_edge_attrs, all_cell_shit_vector],
            torch.tensor(positions.shape[0]),
        )
        edge_attrs = edge_attrs_cell_shift_vector[0]
        cell_shift_vector = edge_attrs_cell_shift_vector[1]

        # check that initial edges are contained in all nbhd edges.
        assert edge_indices.shape == nbhd_edge_indices.shape
    elif (
        (initial_edge_indices is not None)
        and (initial_edge_attrs is not None)
        and (len(initial_edge_attrs.shape) == 1)
    ):
        edge_indices = nbhd_edge_indices.T
        edge_attrs = initial_edge_attrs
        cell_shift_vector = nbhd_cell_shift_vector
    else:
        edge_indices = nbhd_edge_indices.T
        edge_attrs = None
        cell_shift_vector = nbhd_cell_shift_vector

    return edge_indices, edge_attrs, cell_shift_vector
