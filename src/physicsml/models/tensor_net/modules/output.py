from typing import Dict, List

import torch
from torch_geometric.utils import scatter

from physicsml.models.tensor_net.modules.embedding import MLP
from physicsml.models.tensor_net.modules.utils import (
    decompose_irrep_tensor,
    tensor_norm,
)


class NodeScalarOutput(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        mlp_hidden_dims: List[int],
        num_tasks: int,
    ):
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(3 * num_features)
        self.mlp = MLP(
            c_in=3 * num_features,
            hidden_dims=mlp_hidden_dims,
            c_out=num_tasks,
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        X_i = data["X_i"]

        # [num_nodes, num_feats, 3, 3]
        I_i, A_i, S_i = decompose_irrep_tensor(X_i)

        # [num_nodes, num_feats]
        I_norm = tensor_norm(I_i)
        A_norm = tensor_norm(A_i)
        S_norm = tensor_norm(S_i)

        out_i: torch.Tensor = self.mlp(
            self.layer_norm(torch.cat([I_norm, A_norm, S_norm], dim=-1)),
        )

        return out_i


class ScalarOutput(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        mlp_hidden_dims: List[int],
        num_tasks: int,
    ):
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(3 * num_features)
        self.mlp = MLP(
            c_in=3 * num_features,
            hidden_dims=mlp_hidden_dims,
            c_out=num_tasks,
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        X_i = data["X_i"]

        # [num_nodes, num_feats, 3, 3]
        I_i, A_i, S_i = decompose_irrep_tensor(X_i)

        # [num_nodes, num_feats]
        I_norm = tensor_norm(I_i)
        A_norm = tensor_norm(A_i)
        S_norm = tensor_norm(S_i)

        out_i = self.mlp(self.layer_norm(torch.cat([I_norm, A_norm, S_norm], dim=-1)))

        if out_i.shape[1] == 1:
            out: torch.Tensor = scatter(
                src=out_i.squeeze(),
                index=data["batch"],
                dim=0,
                dim_size=int(data["num_graphs"]),
            )
            out = out.unsqueeze(-1)
        else:
            out = scatter(
                src=out_i,
                index=data["batch"],
                dim=0,
                dim_size=int(data["num_graphs"]),
            )

        return out
