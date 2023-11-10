from typing import Optional, Tuple

import torch


def vector_to_skewtensor(vector: torch.Tensor) -> torch.Tensor:
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    skewtensor = torch.stack(
        (
            zero,
            vector[:, 2],
            -vector[:, 1],
            -vector[:, 2],
            zero,
            vector[:, 0],
            vector[:, 1],
            -vector[:, 0],
            zero,
        ),
        dim=1,
    )
    skewtensor = skewtensor.view(-1, 3, 3)
    return skewtensor.squeeze(0)


def vector_to_symtensor(vector: torch.Tensor) -> torch.Tensor:
    outer_tensor = vector.unsqueeze(-1) * vector.unsqueeze(-2)

    trace = outer_tensor.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
    Id = torch.eye(3, device=vector.device, dtype=vector.dtype).unsqueeze(0)

    symtensor = outer_tensor - (1.0 / 3) * trace * Id

    return symtensor


def compose_irrep_tensor(
    scalar_component: torch.Tensor,
    vector_component: torch.Tensor,
    coef_I: Optional[torch.Tensor] = None,
    coef_A: Optional[torch.Tensor] = None,
    coef_S: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    creates an irrep tensor as X = I + A + S, where

    I = scalar * Id
    A = skew_sym(vector)
    S = vector \\cross vector^T - 1/3 Tr(vector \\cross vector^T) Id

    if coefs are specified, it returns

    coef_I * I + coef_A * A + coef_S * S

    Args:
        scalar_component: (*, 1)
        vector_component: (*, 3)
        coef_I: (*, 1)
        coef_A: (*, 1)
        coef_S: (*, 1)

    Returns:

    """

    I = scalar_component.unsqueeze(-1) * torch.eye(  # noqa: E741
        3,
        dtype=scalar_component.dtype,
        device=scalar_component.device,
    ).unsqueeze(0)
    A = vector_to_skewtensor(vector_component)
    S = vector_to_symtensor(vector_component)

    if coef_I is not None:
        I = coef_I * I  # noqa: E741

    if coef_A is not None:
        A = coef_A * A

    if coef_S is not None:
        S = coef_S * S

    return I + A + S


def decompose_irrep_tensor(
    irrep_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decomposes an irrep tensor X into I, A, S
    Args:
        irrep_tensor:

    Returns:

    """

    trace = irrep_tensor.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
    Id = (
        torch.eye(3, device=irrep_tensor.device, dtype=irrep_tensor.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    I = (1.0 / 3) * trace * Id  # noqa: E741
    A = 0.5 * (irrep_tensor - irrep_tensor.transpose(-1, -2))
    S = 0.5 * (irrep_tensor + irrep_tensor.transpose(-1, -2)) - I

    return I, A, S


def tensor_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Computes (wrong) Frobenius norm. BUT!!! This is actually wrong... It should be matmul.
    but this is what they use in the paper.
    """
    norm: torch.Tensor = (x**2).sum((-2, -1))
    return norm


def frobenius_tensor_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Computes Frobenius norm. BUT!!! Different from what they do in the paper.
    """
    norm: torch.Tensor = (
        torch.matmul(x, x.transpose(2, 3)).diagonal(dim1=-1, dim2=-2).sum(-1)
    )
    return norm
