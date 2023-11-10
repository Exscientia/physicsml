from typing import Dict, List, Optional

import torch

from physicsml.models.egnn.egnn_utils import EGNNBlock
from physicsml.models.utils import _AVAILABLE_ACT_FUNCS, make_mlp


class Adapter(torch.nn.Module):
    """Adapter module"""

    def __init__(
        self,
        c_hidden: int,
        ratio_adapter_down: int,
        mlp_activation: Optional[str],
    ) -> None:
        super().__init__()

        list_modules: List[torch.nn.Module] = []
        list_modules.append(torch.nn.Linear(c_hidden, c_hidden // ratio_adapter_down))

        if mlp_activation is not None:
            list_modules.append(_AVAILABLE_ACT_FUNCS[mlp_activation]())
        else:
            list_modules.append(_AVAILABLE_ACT_FUNCS["SiLU"]())
        list_modules.append(torch.nn.Linear(c_hidden // ratio_adapter_down, c_hidden))
        list_modules.append(torch.nn.BatchNorm1d(c_hidden))

        self.adapter_module = torch.nn.Sequential(*list_modules)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        adapted_embedding: torch.Tensor = self.adapter_module(embedding)
        return adapted_embedding


class AdapterEGNN(torch.nn.Module):
    """
    Class for adapter egnn model
    """

    def __init__(
        self,
        num_node_feats: int,
        num_edge_feats: int,
        num_layers: int,
        num_layers_phi: int,
        c_hidden: int,
        ratio_adapter_down: int,
        initial_s: float,
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

        self.adapters_before = torch.nn.ModuleList()
        self.adapters_after = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.s_params_before = torch.nn.ParameterList()
        self.s_params_after = torch.nn.ParameterList()

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
        self.batch_norms.append(torch.nn.BatchNorm1d(c_hidden))

        self.adapters_before.append(
            Adapter(
                c_hidden=c_hidden,
                ratio_adapter_down=ratio_adapter_down,
                mlp_activation=mlp_activation,
            ),
        )
        self.s_params_before.append(torch.nn.Parameter(torch.tensor(initial_s)))
        self.adapters_after.append(
            Adapter(
                c_hidden=c_hidden,
                ratio_adapter_down=ratio_adapter_down,
                mlp_activation=mlp_activation,
            ),
        )
        self.s_params_after.append(torch.nn.Parameter(torch.tensor(initial_s)))

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

            self.batch_norms.append(torch.nn.BatchNorm1d(c_hidden))
            self.adapters_before.append(
                Adapter(
                    c_hidden=c_hidden,
                    ratio_adapter_down=ratio_adapter_down,
                    mlp_activation=mlp_activation,
                ),
            )
            self.s_params_before.append(torch.nn.Parameter(torch.tensor(initial_s)))
            self.adapters_after.append(
                Adapter(
                    c_hidden=c_hidden,
                    ratio_adapter_down=ratio_adapter_down,
                    mlp_activation=mlp_activation,
                ),
            )
            self.s_params_after.append(torch.nn.Parameter(torch.tensor(initial_s)))

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """

        Args:
            data: dict

        Returns: dict

        """

        data["node_feats"] = self.embed_nodes(data["node_attrs"])
        if self.embed_edges is not None:
            data["edge_feats"] = self.embed_edges(data["edge_attrs"])

        ada_node_feats_before = self.adapters_before[0](data["node_feats"])
        data = self.egnn_embedding_block(data)
        ada_node_feats_after = self.adapters_after[0](data["node_feats"])
        data["node_feats"] = (
            self.batch_norms[0](data["node_feats"])
            + self.s_params_before[0] * ada_node_feats_before
            + self.s_params_after[0] * ada_node_feats_after
        )

        for _layer_idx, layer in enumerate(self.egnn_blocks):
            ada_node_feats_before = self.adapters_before[1 + _layer_idx](
                data["node_feats"],
            )
            data = layer(data)
            ada_node_feats_after = self.adapters_after[1 + _layer_idx](
                data["node_feats"],
            )
            data["node_feats"] = (
                self.batch_norms[1 + _layer_idx](data["node_feats"])
                + self.s_params_before[1 + _layer_idx] * ada_node_feats_before
                + self.s_params_after[1 + _layer_idx] * ada_node_feats_after
            )

        return data
