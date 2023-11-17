from typing import Any, Dict, Optional

import torch
from e3nn import o3

from physicsml.lightning.losses.construct_loss import construct_loss
from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.models.allegro.mean_var.default_configs import (
    MeanVarAllegroModelConfig,
)
from physicsml.models.allegro.modules.allegro import (
    Allegro,
    PooledReadoutHead,
    ReadoutHead,
)


class PooledMeanVarAllegroModule(PhysicsMLModuleBase):
    """
    Class for pooled allegro model
    """

    model_config: MeanVarAllegroModelConfig

    def __init__(
        self,
        model_config: MeanVarAllegroModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config=model_config)

        self.losses = self.configure_losses()

        self.compute_forces = model_config.compute_forces
        self.scaling_mean = model_config.scaling_mean
        self.scaling_std = model_config.scaling_std

        self.allegro = Allegro(
            num_node_feats=model_config.num_node_feats,
            num_edge_feats=model_config.num_edge_feats,
            num_layers=model_config.num_layers,
            max_ell=model_config.max_ell,
            num_bessel=model_config.num_bessel,
            bessel_basis_trainable=model_config.bessel_basis_trainable,
            avg_num_neighbours=model_config.avg_num_neighbours or 1.0,
            parity=model_config.parity,
            num_polynomial_cutoff=model_config.num_polynomial_cutoff,
            cut_off=model_config.datamodule.cut_off,
            latent_mlp_latent_dimensions=model_config.latent_mlp_latent_dimensions,
            two_body_latent_mlp_latent_dimensions=model_config.two_body_latent_mlp_latent_dimensions,
            env_embed_mlp_latent_dimensions=model_config.env_embed_mlp_latent_dimensions,
            env_embed_multiplicity=model_config.env_embed_multiplicity,
            embed_initial_edge=model_config.embed_initial_edge,
            per_layer_cutoffs=model_config.per_layer_cutoffs,
            latent_resnet=model_config.latent_resnet,
            latent_resnet_update_ratios=model_config.latent_resnet_update_ratios,
            latent_resnet_update_ratios_learnable=model_config.latent_resnet_update_ratios_learnable,
            sparse_mode=model_config.sparse_mode,
        )

        if model_config.datamodule.y_node_scalars is not None:
            self.node_scalars_head: Optional[torch.nn.Module] = ReadoutHead(
                irrreps_in=self.allegro.irreps_layer_out,
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                mlp_latent_dimensions=model_config.mlp_latent_dimensions,
                num_tasks=len(model_config.datamodule.y_node_scalars),
                avg_num_neighbours=model_config.avg_num_neighbours or 1.0,
                scaling_std=model_config.scaling_std,
                scaling_mean=model_config.scaling_mean,
            )
        else:
            self.node_scalars_head = None

        if model_config.datamodule.y_graph_scalars is not None:
            self.graph_scalars_head: torch.nn.Module = PooledReadoutHead(
                irrreps_in=self.allegro.irreps_layer_out,
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                mlp_latent_dimensions=model_config.mlp_latent_dimensions,
                num_tasks=len(model_config.datamodule.y_graph_scalars),
                avg_num_neighbours=model_config.avg_num_neighbours or 1.0,
                scaling_std=model_config.scaling_std,
                scaling_mean=model_config.scaling_mean,
            )
            self.graph_scalars_head_std: torch.nn.Module = PooledReadoutHead(
                irrreps_in=self.allegro.irreps_layer_out,
                mlp_irreps=o3.Irreps(model_config.mlp_irreps),
                mlp_latent_dimensions=model_config.mlp_latent_dimensions,
                num_tasks=len(model_config.datamodule.y_graph_scalars),
                avg_num_neighbours=model_config.avg_num_neighbours or 1.0,
                scaling_std=model_config.scaling_std,
                scaling_mean=0,
            )
        else:
            raise ValueError("Must specify 'y_graph_scalars' for mean var models")

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data = self.allegro(data)

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

        if self.node_scalars_head is not None:
            node_outputs = self.node_scalars_head(data)
            output["y_node_scalars"] = node_outputs

        return output

    def compute_loss(self, input: Any, target: Any) -> torch.Tensor:
        total_loss: torch.Tensor = torch.zeros(1, device=self.device)
        for y_key, loss in self.losses.items():
            if y_key == "y_graph_scalars":
                total_loss += loss(
                    input["y_graph_scalars"],
                    target["y_graph_scalars"],
                    input["y_graph_scalars::std"] ** 2,
                )
            elif target.get(y_key, None) is not None:
                total_loss += loss(input, target)

        return total_loss

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

        losses["y_graph_scalars"] = torch.nn.GaussianNLLLoss(eps=0.01)

        return losses
