from typing import Dict, Optional, Tuple

import torch
from torch_geometric.utils import scatter


def scatter_mean_std(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="mean")

    std = torch.sqrt(
        scatter(
            src=(src - mean[index]) ** 2,
            index=index,
            dim=dim,
            dim_size=dim_size,
            reduce="mean",
        ),
    )

    return mean, std


EPS = 1e-8


def remap_values(
    old_values: torch.Tensor,
    new_values: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    A function that maps the values in x from the old values to the new values. Be careful to make sure
    that all the values in x are in the old_values tensor and that the old and new values match shape.
    """
    index = torch.bucketize(x.ravel(), old_values)
    return new_values[index].reshape(x.shape)


def clean_up_batch(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    batch_tensor: torch.Tensor,
    num_graphs: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """
    A function to clean up the batch for barlow twins loss. This removes any embeddings which can result in
    zero corr matrix. These embeddings result in zero graphs corr matrices which in turn result in
    nans in the gradients and must be removed.
    """

    # compute the std of the embedding per graph
    mean_z_a, std_z_a = scatter_mean_std(
        src=z_a,
        index=batch_tensor,
        dim=0,
        dim_size=int(num_graphs),
    )
    mean_z_b, std_z_b = scatter_mean_std(
        src=z_b,
        index=batch_tensor,
        dim=0,
        dim_size=int(num_graphs),
    )

    z_a_norm = torch.where(
        std_z_a[batch_tensor] != 0,
        (z_a - mean_z_a[batch_tensor]) / std_z_a[batch_tensor],
        torch.zeros_like(z_a),
    )
    z_b_norm = torch.where(
        std_z_b[batch_tensor] != 0,
        (z_b - mean_z_b[batch_tensor]) / std_z_b[batch_tensor],
        torch.zeros_like(z_b),
    )

    # compute correlation matrix (first aggregate into graphs).
    # shape [num_graphs, dim, dim]
    graph_corr_matrix = scatter(
        src=z_a_norm.unsqueeze(1) * z_b_norm.unsqueeze(2),
        index=batch_tensor,
        dim=0,
        dim_size=int(num_graphs),
        reduce="mean",
    )

    # find the mask where the std is non-zero in all dimensions. These are the embeddings allowed in the loss
    mask = ~(graph_corr_matrix == 0).all(2).all(1)

    if mask.prod() == 0:
        # select these embeddings
        z_a = z_a[mask[batch_tensor]]
        z_b = z_b[mask[batch_tensor]]

        # find the indices of the allowed embeddings
        nonzero_idxs = mask.nonzero().squeeze()

        # find the new batch tensor and num graphs

        ## select masked batch tensor
        masked_batch = batch_tensor[mask[batch_tensor]]

        ## remap values to be incremental
        batch_tensor = remap_values(
            old_values=nonzero_idxs,
            new_values=torch.arange(len(nonzero_idxs), device=nonzero_idxs.device),
            x=masked_batch,
        )
        if batch_tensor.numel() > 0:
            num_graphs = batch_tensor.max() + 1
        else:
            num_graphs = num_graphs * 0

    return z_a, z_b, batch_tensor, num_graphs


def graph_barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    """
    This implements the graph barlow twins loss for graph level info.
    Args:
        z_a: graph embedding of a
        z_b: graph embedding of b
        data: initial data

    Returns:

    """

    feature_dim = z_a.size(1)

    # normalising embeddings by batch. shape [num_graphs, dim]
    std_z_a = z_a.std(0, correction=1)
    std_z_b = z_b.std(0, correction=1)
    z_a_norm = torch.where(
        std_z_a != 0,
        (z_a - z_a.mean(0)) / std_z_a,
        torch.zeros_like(z_a),
    )
    z_b_norm = torch.where(
        std_z_b != 0,
        (z_b - z_b.mean(0)) / std_z_b,
        torch.zeros_like(z_b),
    )

    # computing corr matrix. shape [dim, dim]
    batch_corr_matrix = (z_a_norm.unsqueeze(1) * z_b_norm.unsqueeze(2)).mean(0)

    # find diagonals and off diagonals
    diagonal_mask = torch.eye(feature_dim).to(batch_corr_matrix.device)
    batch_corr_matrix_ii = batch_corr_matrix * diagonal_mask
    batch_corr_matrix_ij = batch_corr_matrix * (1 - diagonal_mask)

    # compute batch loss
    loss_ii = (diagonal_mask - batch_corr_matrix_ii).pow(2).sum((0, 1))
    loss_ij = batch_corr_matrix_ij.pow(2).sum((0, 1)) / feature_dim

    loss: torch.Tensor = loss_ii + loss_ij

    return loss


def node_barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    data: Dict,
) -> torch.Tensor:
    """
    This implements the graph barlow twins loss for node level info. This is not the same as the one
    in the paper. There they aggregate over the whole batch.
    Args:
        z_a: node embedding of a
        z_b: node embedding of b
        data: initial data

    Returns:

    """

    feature_dim = z_a.size(1)

    # clean up
    z_a, z_b, batch_tensor, num_graphs = clean_up_batch(
        z_a=z_a,
        z_b=z_b,
        batch_tensor=data["batch"],
        num_graphs=data["num_graphs"],
    )

    # if no graphs left for loss computation after clean up, then return zero loss
    if num_graphs == 0:
        return torch.tensor(0.0, requires_grad=True, device=num_graphs.device)

    # normalising embeddings by graph. shape [num_nodes, dim]
    mean_z_a, std_z_a = scatter_mean_std(
        src=z_a,
        index=batch_tensor,
        dim=0,
        dim_size=int(num_graphs),
    )
    mean_z_b, std_z_b = scatter_mean_std(
        src=z_b,
        index=batch_tensor,
        dim=0,
        dim_size=int(num_graphs),
    )

    z_a_norm = torch.where(
        std_z_a[batch_tensor] != 0,
        (z_a - mean_z_a[batch_tensor]) / std_z_a[batch_tensor],
        torch.zeros_like(z_a),
    )
    z_b_norm = torch.where(
        std_z_b[batch_tensor] != 0,
        (z_b - mean_z_b[batch_tensor]) / std_z_b[batch_tensor],
        torch.zeros_like(z_b),
    )

    # compute correlation matrix (first aggregate into graphs).
    # shape [num_graphs, dim, dim]
    graph_corr_matrix = scatter(
        src=z_a_norm.unsqueeze(1) * z_b_norm.unsqueeze(2),
        index=batch_tensor,
        dim=0,
        dim_size=int(num_graphs),
        reduce="mean",
    )

    # find diagonals and off diagonals
    diagonal_mask = torch.eye(feature_dim).unsqueeze(0).to(graph_corr_matrix.device)
    graph_corr_matrix_ii = graph_corr_matrix * diagonal_mask
    graph_corr_matrix_ij = graph_corr_matrix * (1 - diagonal_mask)

    # compute loss per graph
    loss_per_graph_ii = (diagonal_mask - graph_corr_matrix_ii).pow(2).sum((1, 2))
    loss_per_graph_ij = graph_corr_matrix_ij.pow(2).sum((1, 2)) / feature_dim

    # total loss average over graphs
    loss_per_graph = loss_per_graph_ii + loss_per_graph_ij

    # mean loss of all graphs
    loss: torch.Tensor = loss_per_graph.mean(0)

    return loss


def edge_barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    data: Dict,
) -> torch.Tensor:
    """
    This implements the graph barlow twins loss for edge level info. This is not the same as the one
    in the paper. There they aggregate over the whole batch. (they also dont do this for edges in the paper).
    Args:
        z_a: edge embedding of a
        z_b: edge embedding of b
        data: initial data

    Returns:

    """

    feature_dim = z_a.size(1)

    # normalising embeddings by graph. shape [num_edges, dim]
    # find num edges by checking num edges between graph max and min indices
    # assumes that graphs are consecutive in the batches!!!
    up = (
        data["edge_index"].unsqueeze(2) < data["ptr"][1:].unsqueeze(0).unsqueeze(1)
    ).all(0)
    down = (
        data["edge_index"].unsqueeze(2) >= data["ptr"][:-1].unsqueeze(0).unsqueeze(1)
    ).all(0)
    edge_graph_membership = up * down
    edge_batch = edge_graph_membership.nonzero()[:, 1]

    # clean up
    z_a, z_b, edge_batch, num_graphs = clean_up_batch(
        z_a=z_a,
        z_b=z_b,
        batch_tensor=edge_batch,
        num_graphs=data["num_graphs"],
    )

    # if no graphs left for loss computation after clean up, then return zero loss
    if num_graphs == 0:
        return torch.tensor(0.0, requires_grad=True, device=num_graphs.device)

    # normalising embeddings by graph. shape [num_nodes, dim]
    mean_z_a, std_z_a = scatter_mean_std(
        src=z_a,
        index=edge_batch,
        dim=0,
        dim_size=int(num_graphs),
    )
    mean_z_b, std_z_b = scatter_mean_std(
        src=z_b,
        index=edge_batch,
        dim=0,
        dim_size=int(num_graphs),
    )

    z_a_norm = torch.where(
        std_z_a[edge_batch] != 0,
        (z_a - mean_z_a[edge_batch]) / std_z_a[edge_batch],
        torch.zeros_like(z_a),
    )
    z_b_norm = torch.where(
        std_z_b[edge_batch] != 0,
        (z_b - mean_z_b[edge_batch]) / std_z_b[edge_batch],
        torch.zeros_like(z_b),
    )

    # compute correlation matrix (first aggregate into nodes then into graphs).
    # shape [num_graphs, dim, dim]
    graph_corr_matrix = scatter(
        src=z_a_norm.unsqueeze(1) * z_b_norm.unsqueeze(2),
        index=edge_batch,
        dim=0,
        dim_size=int(num_graphs),
        reduce="mean",
    )

    # find diagonals and off diagonals
    diagonal_mask = torch.eye(feature_dim).unsqueeze(0).to(graph_corr_matrix.device)
    graph_corr_matrix_ii = graph_corr_matrix * diagonal_mask
    graph_corr_matrix_ij = graph_corr_matrix * (1 - diagonal_mask)

    # compute loss per graph
    loss_per_graph_ii = (diagonal_mask - graph_corr_matrix_ii).pow(2).sum((1, 2))
    loss_per_graph_ij = graph_corr_matrix_ij.pow(2).sum((1, 2)) / feature_dim

    # total loss average over graphs
    loss_per_graph = loss_per_graph_ii + loss_per_graph_ij

    # mean loss of all graphs
    loss: torch.Tensor = loss_per_graph.mean(0)

    return loss
