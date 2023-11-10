from typing import Dict, Optional

import torch

from physicsml.models.egnn.egnn_utils import EGNNBlock
from physicsml.models.utils import make_mlp


class SSF(torch.nn.Module):
    """SSF module"""

    def __init__(self, c_hidden: int) -> None:
        super().__init__()

        self.gamma = torch.nn.Parameter(torch.ones(c_hidden).unsqueeze(0))
        self.beta = torch.nn.Parameter(torch.zeros(c_hidden).unsqueeze(0))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.gamma * embedding + self.beta


class SSFEGNN(torch.nn.Module):
    """
    Class for ssf egnn model
    """

    def __init__(
        self,
        num_node_feats: int,
        num_edge_feats: int,
        num_layers: int,
        num_layers_phi: int,
        c_hidden: int,
        dropout: Optional[float],
        mlp_activation: Optional[str],
        mlp_output_activation: Optional[str],
        num_rbf: int,
        modify_coords: bool,
        bessel_cut_off: float,
    ) -> None:
        super().__init__()

        # node embedding
        self.embed_nodes = make_mlp(
            c_in=num_node_feats,
            c_hidden=c_hidden,
            c_out=c_hidden,
            num_layers=2,
            dropout=dropout,
            mlp_activation=mlp_activation,
            output_activation=mlp_output_activation,
        )

        # edge embedding (if there are edge feats)
        edge_feats_bool = num_edge_feats > 0
        if edge_feats_bool:
            self.embed_edges = make_mlp(
                c_in=num_edge_feats,
                c_hidden=c_hidden,
                c_out=c_hidden,
                num_layers=2,
                dropout=dropout,
                mlp_activation=mlp_activation,
                output_activation=mlp_output_activation,
            )
        else:
            self.embed_edges = None  # type: ignore

        self.ssf = torch.nn.ModuleList()

        self.egnn_embedding_block = EGNNBlock(
            num_layers_phi=num_layers_phi,
            c_hidden=c_hidden,
            edge_feats_bool=edge_feats_bool,
            dropout=dropout,
            mlp_activation=mlp_activation,
            mlp_output_activation=mlp_output_activation,
            modify_coords=modify_coords,
            num_rbf=num_rbf,
            bessel_cut_off=bessel_cut_off,
        )
        self.ssf.append(SSF(c_hidden=c_hidden))

        self.egnn_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.egnn_blocks.append(
                EGNNBlock(
                    num_layers_phi=num_layers_phi,
                    c_hidden=c_hidden,
                    dropout=dropout,
                    mlp_activation=mlp_activation,
                    mlp_output_activation=mlp_output_activation,
                    modify_coords=modify_coords,
                    num_rbf=num_rbf,
                    bessel_cut_off=bessel_cut_off,
                    edge_feats_bool=edge_feats_bool,
                ),
            )

            self.ssf.append(SSF(c_hidden=c_hidden))

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """

        Args:
            data: dict

        Returns: dict

        """

        data["node_feats"] = self.embed_nodes(data["node_attrs"])
        if self.embed_edges is not None:
            data["edge_feats"] = self.embed_edges(data["edge_attrs"])

        data = self.egnn_embedding_block(data)
        data["node_feats"] = self.ssf[0](data["node_feats"])

        for _layer_idx, layer in enumerate(self.egnn_blocks):
            data = layer(data)
            data["node_feats"] = self.ssf[1 + _layer_idx](data["node_feats"])

        return data
