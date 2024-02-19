from typing import Any, Dict, Optional

import torch

from physicsml.lightning.losses.construct_loss import construct_loss
from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.models.egnn.egnn_utils import EGNN, PoolingHead
from physicsml.models.egnn.mean_var.default_configs import MeanVarEGNNModelConfig
from physicsml.models.utils import make_mlp


class PooledMeanVarEGNNModule(PhysicsMLModuleBase):
    """
    Class for pooled egnn model
    """

    model_config: MeanVarEGNNModelConfig

    def __init__(
        self,
        model_config: MeanVarEGNNModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config=model_config)

        self.losses = self.configure_losses()

        self.compute_forces = model_config.compute_forces
        self.scaling_mean = model_config.scaling_mean
        self.scaling_std = model_config.scaling_std

        self.egnn = EGNN(
            num_node_feats=model_config.num_node_feats,
            num_edge_feats=model_config.num_edge_feats,
            num_layers=model_config.num_layers,
            num_layers_phi=model_config.num_layers_phi,
            c_hidden=model_config.c_hidden,
            dropout=model_config.dropout,
            mlp_activation=model_config.mlp_activation,
            mlp_output_activation=model_config.mlp_output_activation,
            num_rbf=model_config.num_rbf,
            modify_coords=model_config.modify_coords,
            bessel_cut_off=model_config.datamodule.cut_off,
        )

        if model_config.datamodule.y_node_scalars is not None:
            self.node_mlp: Optional[torch.nn.Module] = make_mlp(
                c_in=model_config.c_hidden,
                c_hidden=model_config.c_hidden,
                c_out=len(model_config.datamodule.y_node_scalars),
                num_layers=model_config.num_layers_phi,
                dropout=model_config.dropout,
                mlp_activation=model_config.mlp_activation,
                output_activation=model_config.mlp_output_activation,
            )
        else:
            self.node_mlp = None

        if model_config.datamodule.y_edge_scalars is not None:
            self.edge_mlp: Optional[torch.nn.Module] = make_mlp(
                c_in=model_config.c_hidden,
                c_hidden=model_config.c_hidden,
                c_out=len(model_config.datamodule.y_edge_scalars),
                num_layers=model_config.num_layers_phi,
                dropout=model_config.dropout,
                mlp_activation=model_config.mlp_activation,
                output_activation=model_config.mlp_output_activation,
            )
        else:
            self.edge_mlp = None

        if model_config.datamodule.y_graph_scalars is not None:
            self.pooling_head: torch.nn.Module = PoolingHead(
                c_hidden=model_config.c_hidden,
                num_layers_phi=model_config.num_layers_pooling,
                pool_type=model_config.pool_type,
                pool_from=model_config.pool_from,
                num_tasks=len(model_config.datamodule.y_graph_scalars),
                dropout=model_config.dropout,
                mlp_activation=model_config.mlp_activation,
                mlp_output_activation=model_config.mlp_output_activation,
                output_activation=model_config.output_activation,
            )
            self.pooling_head_std: torch.nn.Module = PoolingHead(
                c_hidden=model_config.c_hidden,
                num_layers_phi=model_config.num_layers_pooling,
                pool_type="mean",
                pool_from=model_config.pool_from,
                num_tasks=len(model_config.datamodule.y_graph_scalars),
                dropout=model_config.dropout,
                mlp_activation=model_config.mlp_activation,
                mlp_output_activation=model_config.mlp_output_activation,
                output_activation=model_config.output_activation,
            )
        else:
            raise ValueError("Must specify 'y_graph_scalars' for mean var models.")

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        data = self.egnn(data)

        output = {}
        pooled_output: torch.Tensor = (
            self.pooling_head(data) * self.scaling_std + self.scaling_mean
        )
        pooled_output_std: torch.Tensor = (
            torch.abs(self.pooling_head_std(data)) * self.scaling_std
        )

        if "total_atomic_energy" in data:
            pooled_output = pooled_output + data["total_atomic_energy"].unsqueeze(
                -1,
            )

        output["y_graph_scalars"] = pooled_output
        output["y_graph_scalars::std"] = pooled_output_std

        if self.node_mlp is not None:
            node_output: torch.Tensor = (
                self.node_mlp(data["node_feats"]) * self.scaling_std + self.scaling_mean
            )
            output["y_node_scalars"] = node_output

        if self.edge_mlp is not None:
            edge_output: torch.Tensor = (
                self.edge_mlp(data["edge_feats"]) * self.scaling_std + self.scaling_mean
            )
            output["y_edge_scalars"] = edge_output

        return output

    def compute_loss(self, input: Any, target: Any) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor = torch.zeros(1, device=self.device)
        for y_key, loss in self.losses.items():
            if y_key == "y_graph_scalars":
                key_loss = loss(
                    input["y_graph_scalars"],
                    target["y_graph_scalars"],
                    input["y_graph_scalars::std"] ** 2,
                )
                loss_dict[y_key] = key_loss
                total_loss += key_loss
            elif target.get(y_key, None) is not None:
                key_loss = loss(input, target)
                loss_dict[y_key] = key_loss
                total_loss += key_loss

        loss_dict["loss"] = total_loss
        return loss_dict

    def configure_losses(self) -> Any:
        losses: Dict[str, Optional[Any]] = {}

        if self.model_config.y_node_vector_loss_config is not None:
            losses["y_node_vector"] = construct_loss(
                loss_config=self.model_config.y_node_vector_loss_config,
                column_name="y_node_vector",
            )

        if self.model_config.y_node_scalars_loss_config is not None:
            losses["y_node_scalars"] = construct_loss(
                loss_config=self.model_config.y_node_scalars_loss_config,
                column_name="y_node_scalars",
            )

        if self.model_config.y_edge_scalars_loss_config is not None:
            losses["y_edge_scalars"] = construct_loss(
                loss_config=self.model_config.y_edge_scalars_loss_config,
                column_name="y_edge_scalars",
            )

        losses["y_graph_scalars"] = torch.nn.GaussianNLLLoss(eps=0.01)

        return losses
