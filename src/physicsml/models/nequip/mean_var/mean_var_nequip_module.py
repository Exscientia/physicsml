from typing import Any, Dict, Optional

import torch
from e3nn import o3

from physicsml.lightning.losses.construct_loss import construct_loss
from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.models.nequip.mean_var.default_configs import (
    MeanVarNequipModelConfig,
)
from physicsml.models.nequip.modules.nequip import (
    Nequip,
    PooledReadoutHead,
    ReadoutHead,
)


class PooledMeanVarNequipModule(PhysicsMLModuleBase):
    """
    Class for pooled nequip model
    """

    model_config: MeanVarNequipModelConfig

    def __init__(
        self,
        model_config: MeanVarNequipModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config=model_config)

        self.losses = self.configure_losses()

        self.compute_forces = model_config.compute_forces
        self.scaling_mean = model_config.scaling_mean
        self.scaling_std = model_config.scaling_std

        self.nequip = Nequip(
            cut_off=model_config.datamodule.cut_off,
            num_layers=model_config.num_layers,
            max_ell=model_config.max_ell,
            parity=model_config.parity,
            num_features=model_config.num_features,
            num_bessel=model_config.num_bessel,
            bessel_basis_trainable=model_config.bessel_basis_trainable,
            num_polynomial_cutoff=model_config.num_polynomial_cutoff,
            self_connection=model_config.self_connection,
            resnet=model_config.resnet,
            num_node_feats=model_config.num_node_feats,
            num_edge_feats=model_config.num_edge_feats,
            avg_num_neighbours=model_config.avg_num_neighbours or 1.0,
        )

        if model_config.datamodule.y_node_scalars is not None:
            self.node_scalars_head: Optional[torch.nn.Module] = ReadoutHead(
                irrreps_in=self.nequip.out_irreps[-1],
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
            self.graph_scalars_head: torch.nn.Module = PooledReadoutHead(
                irrreps_in=self.nequip.out_irreps[-1],
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                out_irreps=o3.Irreps(
                    f"{len(model_config.datamodule.y_graph_scalars)}x0e",
                ),
                scaling_std=model_config.scaling_std,
                scaling_mean=model_config.scaling_mean,
            )
            self.graph_scalars_head_std: torch.nn.Module = PooledReadoutHead(
                irrreps_in=self.nequip.out_irreps[-1],
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                out_irreps=o3.Irreps(
                    f"{len(model_config.datamodule.y_graph_scalars)}x0e",
                ),
                scaling_std=model_config.scaling_std,
                scaling_mean=0,
            )
        else:
            raise ValueError("Must specify 'y_graph_scalars' for mean var models.")

        if (model_config.datamodule.y_node_vector is not None) and (
            not self.model_config.compute_forces
        ):
            self.node_vector_head: Optional[torch.nn.Module] = ReadoutHead(
                irrreps_in=self.nequip.out_irreps[-1],
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                out_irreps=o3.Irreps("1o"),
                scaling_std=model_config.scaling_std,
                scaling_mean=model_config.scaling_mean,
            )
        else:
            self.node_vector_head = None

        if model_config.datamodule.y_graph_vector is not None:
            self.graph_vector_head: Optional[torch.nn.Module] = PooledReadoutHead(
                irrreps_in=self.nequip.out_irreps[-1],
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                out_irreps=o3.Irreps("1o"),
                scaling_std=model_config.scaling_std,
                scaling_mean=model_config.scaling_mean,
            )
        else:
            self.graph_vector_head = None

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data = self.nequip(data)

        output = {}

        graph_outputs = self.graph_scalars_head(data)
        graph_outputs_std = torch.abs(self.graph_scalars_head_std(data)) / (
            data["ptr"][1:] - data["ptr"][:-1]
        ).unsqueeze(-1)
        if "total_atomic_energy" in data:
            graph_outputs = graph_outputs + data["total_atomic_energy"].unsqueeze(
                -1,
            )

        output["y_graph_scalars"] = graph_outputs
        output["y_graph_scalars::std"] = graph_outputs_std

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

        losses["y_graph_scalars"] = torch.nn.GaussianNLLLoss(eps=0.01)

        return losses
