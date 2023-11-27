from dataclasses import field
from typing import Dict, Literal, Optional

from pydantic.dataclasses import dataclass

from physicsml.lightning.config import ConfigDict, PhysicsMLModelConfig
from physicsml.models.ani.config import ANIDataModuleConfig


@dataclass(config=ConfigDict)
class EnsembleANIModelConfig(PhysicsMLModelConfig):
    which_ani: Literal["ani1", "ani2", "ani_spice"] = "ani1"
    n_models: int = 4
    scaling_mean: float = 0.0
    scaling_std: float = 1.0
    y_graph_scalars_loss_config: Optional[Dict] = None
    y_node_vector_loss_config: Optional[Dict] = None
    datamodule: ANIDataModuleConfig = field(  # type: ignore
        default_factory=ANIDataModuleConfig,
    )
