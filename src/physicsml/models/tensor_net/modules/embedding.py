from typing import Dict, List, Optional

import torch
from torch_geometric.utils import scatter

from physicsml.models.tensor_net.modules.utils import (
    decompose_irrep_tensor,
    tensor_norm,
    vector_to_skewtensor,
    vector_to_symtensor,
)
from physicsml.models.utils import compute_lengths_and_vectors


class RadialEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        cut_off: float,
    ):
        super().__init__()

        mu_k = torch.linspace(float(torch.exp(-torch.tensor(cut_off))), 1.0, num_radial)
        beta_k = ((2 / num_radial) * (1 - torch.exp(-torch.tensor(cut_off)))) ** -2

        self.register_buffer("mu_k", mu_k.unsqueeze(0))
        self.register_buffer("beta_k", beta_k.unsqueeze(0))

    def forward(self, r_ji: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(-self.beta_k * (torch.exp(-r_ji) - self.mu_k) ** 2)  # type: ignore

        return rbf


class MLP(torch.nn.Module):
    def __init__(
        self,
        c_in: int,
        hidden_dims: List[int],
        c_out: int,
    ):
        super().__init__()

        layers: List[torch.nn.Module] = []
        if len(hidden_dims) == 0:
            layers.append(torch.nn.Linear(c_in, c_out))
        else:
            layers.append(torch.nn.Linear(c_in, hidden_dims[0]))
            layers.append(torch.nn.SiLU())

            for idx in range(len(hidden_dims) - 1):
                layers.append(torch.nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
                layers.append(torch.nn.SiLU())

            layers.append(torch.nn.Linear(hidden_dims[-1], c_out))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layers(x)
        return out


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_node_feats: int,
        num_edge_feats: int,
        num_features: int,
        num_radial: int,
        cut_off: float,
        mlp_hidden_dims: List[int],
    ):
        super().__init__()

        self.cut_off = cut_off

        self.node_embedding = torch.nn.Linear(num_node_feats, num_features, bias=False)
        if num_edge_feats > 0:
            self.edge_embedding: Optional[torch.nn.Module] = torch.nn.Linear(
                num_edge_feats,
                num_features,
                bias=False,
            )
        else:
            self.edge_embedding = None

        self.radial_embedding = RadialEmbedding(num_radial=num_radial, cut_off=cut_off)

        if num_edge_feats > 0:
            num_cat_feats = 3 * num_features
        else:
            num_cat_feats = 2 * num_features

        self.linear_node_cat = torch.nn.Linear(num_cat_feats, num_features)
        self.linear_I = torch.nn.Linear(num_radial, num_features)
        self.linear_A = torch.nn.Linear(num_radial, num_features)
        self.linear_S = torch.nn.Linear(num_radial, num_features)

        self.layer_norm = torch.nn.LayerNorm(num_features)
        self.mlp = MLP(
            c_in=num_features,
            hidden_dims=mlp_hidden_dims,
            c_out=3 * num_features,
        )
        self.silu = torch.nn.SiLU()

        self.linear_I_after = torch.nn.Linear(num_features, num_features, bias=False)
        self.linear_A_after = torch.nn.Linear(num_features, num_features, bias=False)
        self.linear_S_after = torch.nn.Linear(num_features, num_features, bias=False)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "cell" in data:
            cell = data["cell"]
            cell_shift_vector = data["cell_shift_vector"]
        else:
            cell = None
            cell_shift_vector = None

        lengths, vectors = compute_lengths_and_vectors(
            positions=data["coordinates"],
            edge_index=data["edge_index"],
            cell=cell,
            cell_shift_vector=cell_shift_vector,
        )

        normed_r_ji = vectors / lengths

        # [1, 3, 3]
        I_ji = torch.eye(
            3,
            dtype=normed_r_ji.dtype,
            device=normed_r_ji.device,
        ).unsqueeze(0)

        # [num_edges, 3, 3]
        A_ji = vector_to_skewtensor(normed_r_ji)

        # [num_edges, 3, 3]
        S_ji = vector_to_symtensor(normed_r_ji)

        # [num_nodes, num_feats]
        node_feats = self.node_embedding(data["node_attrs"])
        if self.edge_embedding is not None:
            edge_feats = self.edge_embedding(data["edge_attrs"])

        sender = data["edge_index"][0]
        receiver = data["edge_index"][1]
        node_feats_i = node_feats[sender]
        node_feats_j = node_feats[receiver]

        # [num_edges, num_feats]
        if self.edge_embedding is not None:
            catted_feats = torch.cat([node_feats_j, node_feats_i, edge_feats], dim=-1)
        else:
            catted_feats = torch.cat([node_feats_j, node_feats_i], dim=-1)

        node_feats_ji = self.linear_node_cat(catted_feats)

        # [num_edges, num_radial]
        radial_feats = self.radial_embedding(lengths)
        data["radial_feats"] = radial_feats

        # [num_edges, 1]
        phi_ji = torch.where(
            lengths < self.cut_off,
            0.5 * (torch.cos(torch.pi * lengths / self.cut_off) + 1),
            torch.zeros_like(lengths),
        )
        data["phi_ji"] = phi_ji

        # [num_edges, num_feats]
        f_0_I = self.linear_I(radial_feats * phi_ji) * phi_ji
        f_0_A = self.linear_A(radial_feats * phi_ji) * phi_ji
        f_0_S = self.linear_S(radial_feats * phi_ji) * phi_ji

        # [num_edges, num_feats, 3, 3]
        X_ji = node_feats_ji.unsqueeze(-1).unsqueeze(
            -1,
        ) * (  # [num_edges, num_feats, 1, 1]
            f_0_I.unsqueeze(-1).unsqueeze(-1)
            * I_ji.unsqueeze(
                1,
            )  # [num_edges, num_feats, 1, 1] * [num_edges, 1, 3, 3]
            + f_0_A.unsqueeze(-1).unsqueeze(-1)
            * A_ji.unsqueeze(
                1,
            )  # [num_edges, num_feats, 1, 1] * [num_edges, 1, 3, 3]
            + f_0_S.unsqueeze(-1).unsqueeze(-1)
            * S_ji.unsqueeze(
                1,
            )  # [num_edges, num_feats, 1, 1] * [num_edges, 1, 3, 3]
        )

        # [num_nodes, num_feats, 3, 3]
        X_i = scatter(
            src=X_ji,
            index=receiver,
            dim=0,
            dim_size=int(data["num_nodes"]),
        )

        # [num_nodes, num_feats]
        X_norm = tensor_norm(X_i)

        # [num_nodes, 3 * num_feats]
        fs = self.silu(self.mlp(self.layer_norm(X_norm)))
        num_features = X_norm.shape[1]
        fs = fs.reshape(fs.shape[0], num_features, 3)

        # [num_nodes, num_feats]
        f_I = fs[:, :, 0]
        f_A = fs[:, :, 1]
        f_S = fs[:, :, 2]

        # [num_nodes, num_feats, 3, 3]
        I_i, A_i, S_i = decompose_irrep_tensor(X_i)

        I_i = self.linear_I_after(I_i.transpose(1, 3)).transpose(1, 3)
        A_i = self.linear_A_after(A_i.transpose(1, 3)).transpose(1, 3)
        S_i = self.linear_S_after(S_i.transpose(1, 3)).transpose(1, 3)

        # [num_nodes, num_feats, 3, 3]
        X_i = (
            f_I.unsqueeze(-1).unsqueeze(-1) * I_i
            + f_A.unsqueeze(-1).unsqueeze(-1) * A_i
            + f_S.unsqueeze(-1).unsqueeze(-1) * S_i
        )

        data["X_i"] = X_i

        return data
