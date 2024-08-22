from typing import Any

import torch
from molflux.modelzoo.models.lightning.module import (
    SingleBatchStepOutput,
)

from physicsml.lightning.losses.construct_loss import construct_loss
from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.models.egnn.egnn_utils import EGNN
from physicsml.models.egnn.multi_graph.default_configs import (
    MultiGraphEGNNModelConfig,
)
from physicsml.models.egnn.multi_graph.utils import (
    InvariantLigandPocketPoolingHead,
    LigandPocketDiffPoolingHead,
    LigandPocketPoolingHead,
)


class PooledMultiGraphEGNNModule(PhysicsMLModuleBase):
    """
    Class for pooled egnn model
    """

    model_config: MultiGraphEGNNModelConfig

    def __init__(
        self,
        model_config: MultiGraphEGNNModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config=model_config)

        self.losses = self.configure_losses()

        self.scaling_mean = model_config.scaling_mean
        self.scaling_std = model_config.scaling_std

        self.jitter = self.model_config.jitter

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

        if self.model_config.pooling_head == "LigandPocketPoolingHead":
            PoolingHead: Any = LigandPocketPoolingHead
        elif self.model_config.pooling_head == "LigandPocketDiffPoolingHead":
            PoolingHead = LigandPocketDiffPoolingHead
        elif self.model_config.pooling_head == "InvariantLigandPocketPoolingHead":
            PoolingHead = InvariantLigandPocketPoolingHead
        else:
            raise RuntimeError("unknown pooling head")

        self.multi_graph_pooling_head: torch.nn.Module = PoolingHead(
            c_hidden=model_config.c_hidden,
            num_layers_phi=model_config.num_layers_pooling,
            pool_type=model_config.pool_type,
            pool_from=model_config.pool_from,
            num_tasks=len(model_config.datamodule.y_graph_scalars),  # type: ignore
            dropout=model_config.dropout,
            mlp_activation=model_config.mlp_activation,
            mlp_output_activation=model_config.mlp_output_activation,
            output_activation=model_config.output_activation,
        )

    def forward(
        self,
        data: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if self.jitter is not None:
            for k, v in data.items():
                data[k]["coordinates"] = v[
                    "coordinates"
                ] + self.jitter * torch.randn_like(
                    v["coordinates"],
                )

        for k, v in data.items():
            data[k] = self.egnn(v)

        output = {}
        pooled_output: torch.Tensor = (
            self.multi_graph_pooling_head(data) * self.scaling_std + self.scaling_mean
        )
        output["y_graph_scalars"] = pooled_output

        return output

    def compute_loss(self, input: Any, target: Any) -> dict[str, torch.Tensor]:
        loss_dict: dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor = torch.zeros(1, device=self.device)
        for y_key, loss in self.losses.items():
            if target.get(y_key, None) is not None:
                key_loss = loss(input, target)
                loss_dict[y_key] = key_loss
                total_loss += key_loss

        loss_dict["loss"] = total_loss
        return loss_dict

    def configure_losses(self) -> Any:
        losses: dict[str, Any | None] = {}

        losses["y_graph_scalars"] = construct_loss(
            loss_config=self.model_config.y_graph_scalars_loss_config,  # type: ignore
            column_name="y_graph_scalars",
        )

        return losses

    def _training_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: str | None,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        """fit method for doing a training step"""

        batch_dict = {}
        for k, v in single_source_batch.items():
            batch_dict[k] = self.graph_batch_to_batch_dict(v)

        output = self(batch_dict)

        loss_dict = self.compute_loss(output, batch_dict[next(iter(batch_dict.keys()))])
        return (
            loss_dict["loss"],
            loss_dict,
            batch_dict[next(iter(batch_dict.keys()))]["ptr"].shape[0] - 1,
        )

    def _validation_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: str | None,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        """fit method for doing a validation step"""

        batch_dict = {}
        for k, v in single_source_batch.items():
            batch_dict[k] = self.graph_batch_to_batch_dict(v)

        output = self(batch_dict)

        loss_dict = self.compute_loss(output, batch_dict[next(iter(batch_dict.keys()))])

        return (
            loss_dict["loss"],
            loss_dict,
            batch_dict[next(iter(batch_dict.keys()))]["ptr"].shape[0] - 1,
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """method for doing a predict step"""

        batch_dict = {}
        for k, v in batch.items():
            batch_dict[k] = self.graph_batch_to_batch_dict(v)

        output = self(batch_dict)

        detached_output: Any
        # if tensor
        if isinstance(output, torch.Tensor):
            detached_output = output.detach()
        # elif dict of tensors
        elif isinstance(output, dict) and all(
            isinstance(x, torch.Tensor) for x in output.values()
        ):
            detached_output = {}
            for k, v in output.items():
                detached_output[k] = v.detach()
        else:
            detached_output = output

        del output
        del batch
        del batch_dict

        return detached_output
