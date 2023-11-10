from dataclasses import field
from typing import Dict, List, Literal, Optional, Tuple

from molflux.modelzoo.models.lightning.config import (
    DataModuleConfig,
    LightningConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from pydantic.dataclasses import dataclass


class ConfigDict:
    extra = "forbid"
    arbitrary_types_allowed = True
    smart_union = True


@dataclass(config=ConfigDict)
class ANIDataModuleConfig(DataModuleConfig):
    y_node_scalars: Optional[List[str]] = None
    y_node_vector: Optional[str] = None
    y_edge_scalars: Optional[List[str]] = None
    y_edge_vector: Optional[str] = None
    y_graph_scalars: Optional[List[str]] = None
    y_graph_vector: Optional[str] = None
    atomic_numbers_col: str = "physicsml_atom_numbers"
    coordinates_col: str = "physicsml_coordinates"
    total_atomic_energy_col: str = "physicsml_total_atomic_energy"
    pbc: Optional[Tuple[bool, bool, bool]] = None
    cell: Optional[List[List[float]]] = None
    pre_batch: Optional[Literal["in_memory", "on_disk"]] = None
    pre_batch_in_memory: bool = False  # TODO: Deprecate
    train_batch_size: Optional[int] = None  # TODO: Deprecate
    validation_batch_size: Optional[int] = None  # TODO: Deprecate


@dataclass(config=ConfigDict)
class ANIModelConfig(LightningConfig):
    which_ani: Literal["ani1", "ani2", "ani_spice"] = "ani1"
    scaling_mean: float = 0.0
    scaling_std: float = 1.0
    y_graph_scalars_loss_config: Optional[Dict] = None
    y_node_vector_loss_config: Optional[Dict] = None
    datamodule: ANIDataModuleConfig = field(
        default_factory=ANIDataModuleConfig,
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(name="Adam", config={"lr": 1e-3}),
    )
    scheduler: Optional[SchedulerConfig] = field(default_factory=SchedulerConfig)
