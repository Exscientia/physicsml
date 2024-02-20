from typing import Dict, List

import torch
from torch_geometric.utils import scatter

from physicsml.models.tensor_net.modules.embedding import MLP
from physicsml.models.tensor_net.modules.utils import (
    decompose_irrep_tensor,
    tensor_norm,
)


class Interaction(torch.nn.Module):
    def __init__(self, num_features: int, num_radial: int, mlp_hidden_dims: List[int]):
        super().__init__()

        self.linear_I = torch.nn.Linear(num_features, num_features, bias=False)
        self.linear_A = torch.nn.Linear(num_features, num_features, bias=False)
        self.linear_S = torch.nn.Linear(num_features, num_features, bias=False)

        self.silu = torch.nn.SiLU()
        self.mlp = MLP(
            c_in=num_radial,
            hidden_dims=mlp_hidden_dims,
            c_out=3 * num_features,
        )

        self.linear_I_new = torch.nn.Linear(num_features, num_features, bias=False)
        self.linear_A_new = torch.nn.Linear(num_features, num_features, bias=False)
        self.linear_S_new = torch.nn.Linear(num_features, num_features, bias=False)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [num_nodes, num_feats, 3, 3]
        X_i = data["X_i"]

        # [num_nodes, num_feats]
        X_norm = tensor_norm(X_i)

        # [num_nodes, num_feats, 3, 3]
        X_i = X_i / (X_norm.unsqueeze(-1).unsqueeze(-1) + 1)

        # [num_nodes, num_feats, 3, 3]
        I_i, A_i, S_i = decompose_irrep_tensor(X_i)

        # [num_nodes, num_feats, 3, 3]
        I_i = self.linear_I(I_i.transpose(1, 3)).transpose(1, 3)
        A_i = self.linear_A(A_i.transpose(1, 3)).transpose(1, 3)
        S_i = self.linear_S(S_i.transpose(1, 3)).transpose(1, 3)

        # [num_nodes, num_feats, 3, 3]
        Y_i = I_i + A_i + S_i

        # [num_edges, 3 * num_feats]
        fs = data["phi_ji"] * self.silu(self.mlp(data["radial_feats"] * data["phi_ji"]))
        num_features = X_norm.shape[1]
        fs = fs.reshape(fs.shape[0], num_features, 3)

        # [num_nodes, num_feats]
        f_I_ji = fs[:, :, 0]
        f_A_ji = fs[:, :, 1]
        f_S_ji = fs[:, :, 2]

        sender = data["edge_index"][0]
        receiver = data["edge_index"][1]

        # [num_edges, num_feats, 3, 3]
        M_ji = (
            f_I_ji.unsqueeze(-1).unsqueeze(-1) * I_i[sender]
            + f_A_ji.unsqueeze(-1).unsqueeze(-1) * A_i[sender]
            + f_S_ji.unsqueeze(-1).unsqueeze(-1) * S_i[sender]
        )

        # [num_nodes, num_feats, 3, 3]
        M_i = scatter(
            src=M_ji,
            index=receiver,
            dim=0,
            dim_size=int(data["num_nodes"]),
        )

        # [num_nodes, num_feats, 3, 3]
        I_new_i, A_new_i, S_new_i = decompose_irrep_tensor(
            torch.matmul(M_i, Y_i) + torch.matmul(Y_i, M_i),
        )

        # [num_nodes, num_feats, 3, 3]
        IAS = I_new_i + A_new_i + S_new_i

        # [num_nodes, num_feats]
        IAS_norm = tensor_norm(IAS)

        # [num_nodes, num_feats, 3, 3]
        I_new_i = I_new_i / (IAS_norm.unsqueeze(-1).unsqueeze(-1) + 1)
        A_new_i = A_new_i / (IAS_norm.unsqueeze(-1).unsqueeze(-1) + 1)
        S_new_i = S_new_i / (IAS_norm.unsqueeze(-1).unsqueeze(-1) + 1)

        # [num_nodes, num_feats, 3, 3]
        I_new_i = self.linear_I_new(I_new_i.transpose(1, 3)).transpose(1, 3)
        A_new_i = self.linear_A_new(A_new_i.transpose(1, 3)).transpose(1, 3)
        S_new_i = self.linear_S_new(S_new_i.transpose(1, 3)).transpose(1, 3)

        Y_i = I_new_i + A_new_i + S_new_i

        data["X_i"] = X_i + Y_i + torch.matmul(Y_i, Y_i)

        return data
