from dataclasses import field
from typing import Dict, List, Optional

from molflux.modelzoo.models.lightning.config import OptimizerConfig, SchedulerConfig
from pydantic.dataclasses import dataclass

from physicsml.lightning.config import ConfigDict, PhysicsMLModelConfig


@dataclass(config=ConfigDict)
class TensorNetModelConfig(PhysicsMLModelConfig):
    num_node_feats: int = 0
    num_edge_feats: int = 0
    num_features: int = 256
    num_radial: int = 64
    num_interaction_layers: int = 3
    embedding_mlp_hidden_dims: List[int] = field(default_factory=lambda: [512])
    interaction_mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 512])
    scalar_output_mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    scaling_mean: float = 0.0
    scaling_std: float = 1.0
    y_node_scalars_loss_config: Optional[Dict] = None
    y_graph_scalars_loss_config: Optional[Dict] = None
    y_node_vector_loss_config: Optional[Dict] = None
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            name="AdamW",
            config={
                "lr": 2e-4,
            },
        ),
    )
    scheduler: Optional[SchedulerConfig] = field(
        default_factory=lambda: SchedulerConfig(
            name="ReduceLROnPlateau",
            config={
                "factor": 0.8,
                "patience": 15,
            },
            monitor="val/total/loss",
            interval="epoch",
        ),
    )
