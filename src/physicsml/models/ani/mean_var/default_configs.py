from dataclasses import field
from typing import Literal

from pydantic.v1 import dataclasses

from physicsml.lightning.config import ConfigDict, PhysicsMLModelConfig
from physicsml.models.ani.config import ANIDataModuleConfig


@dataclasses.dataclass(config=ConfigDict)
class MeanVarANIModelConfig(PhysicsMLModelConfig):
    which_ani: Literal["ani1", "ani2", "ani_spice"] = "ani1"
    scaling_mean: float = 0.0
    scaling_std: float = 1.0
    y_graph_scalars_loss_config: dict | None = None
    y_node_vector_loss_config: dict | None = None
    datamodule: ANIDataModuleConfig = field(  # type: ignore
        default_factory=ANIDataModuleConfig,
    )
