from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

_AVAILABLE_ACT_FUNCS = {
    "ELU": nn.ELU,
    "Hardshrink": nn.Hardshrink,
    "Hardsigmoid": nn.Hardsigmoid,
    "Hardtanh": nn.Hardtanh,
    "Hardswish": nn.Hardswish,
    "LeakyReLU": nn.LeakyReLU,
    "LogSigmoid": nn.LogSigmoid,
    "MultiheadAttention": nn.MultiheadAttention,
    "PReLU": nn.PReLU,
    "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6,
    "RReLU": nn.RReLU,
    "SELU": nn.SELU,
    "CELU": nn.CELU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "SiLU": nn.SiLU,
    "Mish": nn.Mish,
    "Softplus": nn.Softplus,
    "Softshrink": nn.Softshrink,
    "Softsign": nn.Softsign,
    "Tanh": nn.Tanh,
    "Tanhshrink": nn.Tanhshrink,
    "Threshold": nn.Threshold,
    "GLU": nn.GLU,
    "Softmin": nn.Softmin,
    "Softmax": nn.Softmax,
    "Softmax2d": nn.Softmax2d,
    "LogSoftmax": nn.LogSoftmax,
    "AdaptiveLogSoftmaxWithLoss": nn.AdaptiveLogSoftmaxWithLoss,
}


def make_mlp(
    c_in: int,
    c_hidden: int,
    c_out: int,
    num_layers: int,
    dropout: Optional[float] = None,
    mlp_activation: Optional[str] = "SiLU",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    phi_list: List[Any] = []
    phi_list.append(nn.Linear(c_in, c_hidden))
    if mlp_activation is not None:
        phi_list.append(_AVAILABLE_ACT_FUNCS[mlp_activation]())

    for _i in range(num_layers - 2):
        phi_list.append(nn.Linear(c_hidden, c_hidden))
        if mlp_activation is not None:
            phi_list.append(_AVAILABLE_ACT_FUNCS[mlp_activation]())

    if dropout is not None:
        phi_list.append(nn.Dropout(p=dropout))

    phi_list.append(nn.Linear(c_hidden, c_out))

    if output_activation is not None:
        phi_list.append(_AVAILABLE_ACT_FUNCS[output_activation]())

    phi = nn.Sequential(*phi_list)
    return phi


def compute_lengths_and_vectors(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
    cell_shift_vector: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]

    positions_i = positions[sender]
    positions_j = positions[receiver]
    r_ji = positions_j - positions_i

    if (cell is not None) and (cell_shift_vector is not None):
        # have to separate this for a torchscript compiling bug...
        r_ji = (
            r_ji
            + cell_shift_vector[:, 0].unsqueeze(-1) * cell[0].unsqueeze(0)
            + cell_shift_vector[:, 1].unsqueeze(-1) * cell[1].unsqueeze(0)
            + cell_shift_vector[:, 2].unsqueeze(-1) * cell[2].unsqueeze(0)
        )

    abs_r_ji = torch.norm(r_ji, dim=-1).unsqueeze(-1).clamp(min=1e-8)

    return abs_r_ji, r_ji


def generate_random_mask(
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    ratio_masked_nodes: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, counts = batch.unique(return_counts=True)

    cum_sum = 0
    masked_indices_list = []
    for count in counts:
        count_int: int = int(count)
        num_masked_nodes: int = int(ratio_masked_nodes * count_int)
        random_idxs = torch.randperm(count_int)[:num_masked_nodes]
        masked_indices_list.append(random_idxs + cum_sum)
        cum_sum += count_int

    masked_indices = torch.cat(masked_indices_list).to(batch.device)

    node_mask = torch.ones_like(batch).type(torch.bool)
    node_mask[masked_indices] = False

    edge_mask = (
        ~(edge_index.T[:, None, :] == masked_indices[None, :, None]).any(2).any(1)
    )

    return node_mask, edge_mask
