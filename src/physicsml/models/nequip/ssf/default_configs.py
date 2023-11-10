from dataclasses import field
from typing import Dict, Optional

from molflux.modelzoo.models.lightning.config import OptimizerConfig, SchedulerConfig
from pydantic.dataclasses import dataclass

from physicsml.lightning.config import ConfigDict, PhysicsMLModelConfig


@dataclass(config=ConfigDict)
class SSFNequipModelConfig(PhysicsMLModelConfig):
    num_node_feats: int = 0
    num_edge_feats: int = 0
    num_layers: int = 4
    max_ell: int = 2
    parity: bool = True
    num_features: int = 32
    mlp_irreps: str = "16x0e"
    num_bessel: int = 8
    bessel_basis_trainable: bool = True
    num_polynomial_cutoff: int = 6
    self_connection: bool = True
    resnet: bool = True
    avg_num_neighbours: Optional[float] = None
    scaling_mean: float = 0.0
    scaling_std: float = 1.0
    y_node_scalars_loss_config: Optional[Dict] = None
    y_graph_scalars_loss_config: Optional[Dict] = None
    y_node_vector_loss_config: Optional[Dict] = None
    y_graph_vector_loss_config: Optional[Dict] = None
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            name="Adam",
            config={
                "lr": 1e-2,
                "amsgrad": True,
            },
        ),
    )
    scheduler: Optional[SchedulerConfig] = field(
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
