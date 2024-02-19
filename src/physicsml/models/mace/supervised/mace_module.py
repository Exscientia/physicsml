from typing import Any, Dict, Optional

import torch
from e3nn import o3
from molflux.modelzoo.models.lightning.trainer.optimizers.stock_optimizers import (
    AVAILABLE_OPTIMIZERS,
)

from physicsml.lightning.losses.construct_loss import construct_loss
from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.models.mace.modules.mace import MACE, PooledReadoutHead, ReadoutHead
from physicsml.models.mace.supervised.default_configs import MACEModelConfig


class PooledMACEModule(PhysicsMLModuleBase):
    """
    Class for pooled mace model
    """

    model_config: MACEModelConfig

    def __init__(
        self,
        model_config: MACEModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config=model_config)

        self.losses = self.configure_losses()

        self.compute_forces = model_config.compute_forces
        self.scaling_mean = model_config.scaling_mean
        self.scaling_std = model_config.scaling_std

        self.mace = MACE(
            cut_off=model_config.datamodule.cut_off,
            num_bessel=model_config.num_bessel,
            num_polynomial_cutoff=model_config.num_polynomial_cutoff,
            max_ell=model_config.max_ell,
            num_interactions=model_config.num_interactions,
            num_node_feats=model_config.num_node_feats,
            num_edge_feats=model_config.num_edge_feats,
            hidden_irreps=model_config.hidden_irreps,
            avg_num_neighbours=model_config.avg_num_neighbours or 1.0,
            correlation=model_config.correlation,
        )

        if model_config.datamodule.y_node_scalars is not None:
            self.node_scalars_head: Optional[torch.nn.Module] = ReadoutHead(
                list_in_irreps=self.mace.out_irreps,
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                out_irreps=o3.Irreps(
                    f"{len(model_config.datamodule.y_node_scalars)}x0e",
                ),
                scaling_std=model_config.scaling_std,
                scaling_mean=model_config.scaling_mean,
            )
        else:
            self.node_scalars_head = None

        if model_config.datamodule.y_graph_scalars is not None:
            self.graph_scalars_head: Optional[torch.nn.Module] = PooledReadoutHead(
                list_in_irreps=self.mace.out_irreps,
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                out_irreps=o3.Irreps(
                    f"{len(model_config.datamodule.y_graph_scalars)}x0e",
                ),
                scaling_std=model_config.scaling_std,
                scaling_mean=model_config.scaling_mean,
            )
        else:
            self.graph_scalars_head = None

        if (model_config.datamodule.y_node_vector is not None) and (
            not self.model_config.compute_forces
        ):
            self.node_vector_head: Optional[torch.nn.Module] = ReadoutHead(
                list_in_irreps=self.mace.out_irreps,
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                out_irreps=o3.Irreps("1o"),
                scaling_std=model_config.scaling_std,
                scaling_mean=model_config.scaling_mean,
            )
        else:
            self.node_vector_head = None

        if model_config.datamodule.y_graph_vector is not None:
            self.graph_vector_head: Optional[torch.nn.Module] = PooledReadoutHead(
                list_in_irreps=self.mace.out_irreps,
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                out_irreps=o3.Irreps("1o"),
                scaling_std=model_config.scaling_std,
                scaling_mean=model_config.scaling_mean,
            )
        else:
            self.graph_vector_head = None

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data = self.mace(data)

        output = {}

        if self.graph_scalars_head is not None:
            graph_outputs = self.graph_scalars_head(data)
            if "total_atomic_energy" in data:
                graph_outputs = graph_outputs + data["total_atomic_energy"].unsqueeze(
                    -1,
                )

            output["y_graph_scalars"] = graph_outputs

        if self.graph_vector_head is not None:
            graph_vector = self.graph_vector_head(data)
            output["y_graph_vector"] = graph_vector

        if self.node_scalars_head is not None:
            node_outputs = self.node_scalars_head(data)
            output["y_node_scalars"] = node_outputs

        if self.node_vector_head is not None:
            node_vector = self.node_vector_head(data)
            output["y_node_vector"] = node_vector

        return output

    def configure_optimizers(self) -> Dict:  # type: ignore
        """Returns: optimizer and (optional) scheduler"""

        out = {}

        assert self.model_config.optimizer is not None

        # Optimizer
        decay_interactions = []
        for name, _ in self.mace.interactions.named_parameters():
            if "linear.weight" in name or "residual_connection_layer.weight" in name:
                decay_interactions.append(name)

        for name, _ in self.mace.messages.named_parameters():
            if "linear.weight" in name or "residual_connection_layer.weight" in name:
                decay_interactions.append(name)

        for name, _ in self.mace.node_updates.named_parameters():
            if "linear.weight" in name or "residual_connection_layer.weight" in name:
                decay_interactions.append(name)

        if "weight_decay" in self.model_config.optimizer.config:
            weight_decay = self.model_config.optimizer.config.pop("weight_decay")
        else:
            weight_decay = 0.0

        params_list = [
            {
                "name": name,
                "params": param,
                "weight_decay": weight_decay if name in decay_interactions else 0.0,
            }
            for name, param in self.named_parameters()
        ]

        optimizer_config = self.model_config.optimizer
        optimizer = AVAILABLE_OPTIMIZERS[optimizer_config.name](
            params=params_list,
            **optimizer_config.config,
        )
        out["optimizer"] = optimizer

        if self.model_config.scheduler is not None:
            out["lr_scheduler"] = self.model_config.scheduler.prepare_scheduler(
                optimizer=optimizer,
                trainer=self.trainer,
            )

        return out

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

        if self.model_config.y_graph_vector_loss_config is not None:
            losses["y_graph_vector"] = construct_loss(
                loss_config=self.model_config.y_graph_vector_loss_config,
                column_name="y_graph_vector",
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
