from typing import Any, Dict, Optional

import torch

from physicsml.lightning.losses.construct_loss import construct_loss
from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.models.tensor_net.modules.embedding import Embedding
from physicsml.models.tensor_net.modules.interaction import Interaction
from physicsml.models.tensor_net.modules.output import (
    NodeScalarOutput,
    ScalarOutput,
)
from physicsml.models.tensor_net.supervised.default_configs import (
    TensorNetModelConfig,
)


class PooledTensorNetModule(PhysicsMLModuleBase):
    model_config: TensorNetModelConfig

    def __init__(
        self,
        model_config: TensorNetModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config=model_config)

        self.losses = self.configure_losses()

        self.compute_forces = model_config.compute_forces
        self.scaling_mean = model_config.scaling_mean
        self.scaling_std = model_config.scaling_std

        self.embedding = Embedding(
            num_node_feats=model_config.num_node_feats,
            num_edge_feats=model_config.num_edge_feats,
            num_features=model_config.num_features,
            num_radial=model_config.num_radial,
            cut_off=model_config.datamodule.cut_off,
            mlp_hidden_dims=model_config.embedding_mlp_hidden_dims,
        )

        self.interactions = torch.nn.ModuleList()

        for _ in range(model_config.num_interaction_layers):
            self.interactions.append(
                Interaction(
                    num_features=model_config.num_features,
                    num_radial=model_config.num_radial,
                    mlp_hidden_dims=model_config.interaction_mlp_hidden_dims,
                ),
            )

        if model_config.datamodule.y_node_scalars is not None:
            self.node_scalar_output: Optional[NodeScalarOutput] = NodeScalarOutput(
                num_features=model_config.num_features,
                mlp_hidden_dims=model_config.scalar_output_mlp_hidden_dims,
                num_tasks=len(model_config.datamodule.y_node_scalars),
            )
        else:
            self.node_scalar_output = None

        if model_config.datamodule.y_graph_scalars is not None:
            self.scalar_output: Optional[ScalarOutput] = ScalarOutput(
                num_features=model_config.num_features,
                mlp_hidden_dims=model_config.scalar_output_mlp_hidden_dims,
                num_tasks=len(model_config.datamodule.y_graph_scalars),
            )
        else:
            self.scalar_output = None

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data = self.embedding(data)

        for interaction in self.interactions:
            data = interaction(data)

        output = {}
        if self.scalar_output is not None:
            pooled_output: torch.Tensor = (
                self.scalar_output(data) * self.scaling_std + self.scaling_mean
            )

            if "total_atomic_energy" in data:
                pooled_output = pooled_output + data["total_atomic_energy"].unsqueeze(
                    -1,
                )

            output["y_graph_scalars"] = pooled_output

        if self.node_scalar_output is not None:
            node_output: torch.Tensor = (
                self.node_scalar_output(data) * self.scaling_std + self.scaling_mean
            )
            output["y_node_scalars"] = node_output

        return output

    def compute_loss(self, input: Any, target: Any) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor = torch.zeros(1, device=self.device)
        for y_key, loss in self.losses.items():
            if target.get(y_key, None) is not None:
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

        if self.model_config.y_graph_scalars_loss_config is not None:
            losses["y_graph_scalars"] = construct_loss(
                loss_config=self.model_config.y_graph_scalars_loss_config,
                column_name="y_graph_scalars",
            )

        return losses
