from typing import Dict, Optional

import torch
from e3nn import o3

from physicsml.models.nequip.modules.interaction_block import InteractionBlock

from ._gate import Gate


def tp_path_exists(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    ir_out: o3.Irreps,
) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class ConvNetLayer(torch.nn.Module):
    def __init__(
        self,
        interaction_irreps_in: o3.Irreps,
        node_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbours: Optional[float] = None,
        self_connection: bool = True,
        num_layers: int = 3,
        resnet: bool = False,
    ):
        super().__init__()

        self.hidden_irreps = hidden_irreps
        self.num_layers = num_layers

        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if ir.l == 0
                and tp_path_exists(interaction_irreps_in, edge_attrs_irreps, ir)
            ],
        )

        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if ir.l > 0
                and tp_path_exists(interaction_irreps_in, edge_attrs_irreps, ir)
            ],
        )

        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        ir = (
            "0e"
            if tp_path_exists(interaction_irreps_in, edge_attrs_irreps, "0e")
            else "0o"
        )
        irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

        self.equivariant_nonlin = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[
                torch.tanh if ir.p == -1 else torch.nn.SiLU()
                for _, ir in irreps_scalars
            ],
            irreps_gates=irreps_gates,
            act_gates=[
                torch.tanh if ir.p == -1 else torch.nn.SiLU() for _, ir in irreps_gates
            ],
            irreps_gated=irreps_gated,
        )

        self.conv_irreps_out = self.equivariant_nonlin.irreps_in.simplify()

        if irreps_layer_out == interaction_irreps_in and resnet:
            # We are doing resnet updates and can for this layer
            self.resnet = True
        else:
            self.resnet = False

        self.conv = InteractionBlock(
            interaction_irreps_in=interaction_irreps_in,
            interaction_irreps_out=self.conv_irreps_out,
            node_attrs_irreps=node_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            avg_num_neighbours=avg_num_neighbours,
            self_connection=self_connection,
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # save old features for resnet
        old_x = data["node_feats"]
        # run convolution
        data = self.conv(data)
        # do nonlinearity
        data["node_feats"] = self.equivariant_nonlin(data["node_feats"])
        # do resnet
        if self.resnet:
            data["node_feats"] = old_x + data["node_feats"]

        return data
