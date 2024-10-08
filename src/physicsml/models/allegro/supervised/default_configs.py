from dataclasses import field

from molflux.modelzoo.models.lightning.config import OptimizerConfig, SchedulerConfig
from pydantic.v1 import dataclasses

from physicsml.lightning.config import ConfigDict, PhysicsMLModelConfig


@dataclasses.dataclass(config=ConfigDict)
class AllegroModelConfig(PhysicsMLModelConfig):
    num_node_feats: int = 0
    num_edge_feats: int = 0
    num_layers: int = 2
    max_ell: int = 2
    parity: bool = True
    mlp_irreps: str = "16x0e"
    mlp_latent_dimensions: list[int] = field(default_factory=lambda: [128])
    latent_mlp_latent_dimensions: list[int] = field(
        default_factory=lambda: [1024, 1024, 1024],
    )
    env_embed_multiplicity: int = 32
    two_body_latent_mlp_latent_dimensions: list[int] = field(
        default_factory=lambda: [128, 256, 512, 1024],
    )
    env_embed_mlp_latent_dimensions: list[int] = field(default_factory=lambda: [])
    num_bessel: int = 8
    bessel_basis_trainable: bool = True
    num_polynomial_cutoff: int = 6
    avg_num_neighbours: float | None = None
    embed_initial_edge: bool = True
    per_layer_cutoffs: list[float] | None = None
    latent_resnet: bool = True
    latent_resnet_update_ratios: list[float] | None = None
    latent_resnet_update_ratios_learnable: bool = False
    sparse_mode: str | None = None
    scaling_mean: float = 0.0
    scaling_std: float = 1.0
    y_node_scalars_loss_config: dict | None = None
    y_node_vector_loss_config: dict | None = None
    y_graph_scalars_loss_config: dict | None = None
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            name="Adam",
            config={
                "lr": 1e-2,
                "amsgrad": True,
            },
        ),
    )
    scheduler: SchedulerConfig | None = field(
        default_factory=lambda: SchedulerConfig(
            name="ReduceLROnPlateau",
            config={
                "factor": 0.8,
                "patience": 50,
            },
            monitor="val/total/loss",
            interval="epoch",
        ),
    )
