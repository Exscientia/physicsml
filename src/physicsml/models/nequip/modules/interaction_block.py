from typing import Dict, List, Optional

import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from torch_geometric.utils import scatter


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        interaction_irreps_in: o3.Irreps,
        interaction_irreps_out: o3.Irreps,
        node_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        avg_num_neighbours: Optional[float] = None,
        self_connection: bool = True,
    ) -> None:
        super().__init__()

        self.avg_num_neighbours = avg_num_neighbours

        self.linear_1 = o3.Linear(
            irreps_in=interaction_irreps_in,
            irreps_out=interaction_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid_list: List = []
        instructions = []

        for i, (mul, ir_in) in enumerate(interaction_irreps_in):
            for j, (_, ir_edge) in enumerate(edge_attrs_irreps):
                for ir_out in ir_in * ir_edge:
                    if ir_out in interaction_irreps_out:
                        k = len(irreps_mid_list)
                        irreps_mid_list.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = o3.Irreps(irreps_mid_list)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp = o3.TensorProduct(
            interaction_irreps_in,
            edge_attrs_irreps,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = FullyConnectedNet(
            [edge_feats_irreps.num_irreps, 64, 64, self.tp.weight_numel],
            torch.nn.SiLU(),
        )

        self.linear_2 = o3.Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=interaction_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        if self_connection:
            self.self_connection_layer: Optional[
                o3.FullyConnectedTensorProduct
            ] = o3.FullyConnectedTensorProduct(
                interaction_irreps_in,
                node_attrs_irreps,
                interaction_irreps_out,
            )
        else:
            self.self_connection_layer = None

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        weight = self.fc(data["edge_feats"])

        x = data["node_feats"]
        edge_src = data["edge_index"][1]
        edge_dst = data["edge_index"][0]

        if self.self_connection_layer is not None:
            sc = self.self_connection_layer(x, data["node_attrs"])

        x = self.linear_1(x)
        edge_features = self.tp(x[edge_src], data["edge_attrs"], weight)
        x = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

        # Necessary to get TorchScript to be able to type infer when its not None
        if self.avg_num_neighbours is not None:
            x = x.div(self.avg_num_neighbours**0.5)

        x = self.linear_2(x)

        if self.self_connection_layer is not None:
            x = x + sc

        data["node_feats"] = x
        return data
