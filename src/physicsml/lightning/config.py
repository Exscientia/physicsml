from dataclasses import field
from typing import List, Literal, Optional, Tuple

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
class PhysicsMLDataModuleConfig(DataModuleConfig):
    num_elements: int = 0
    y_node_scalars: Optional[List[str]] = None
    y_node_vector: Optional[str] = None
    y_edge_scalars: Optional[List[str]] = None
    y_edge_vector: Optional[str] = None
    y_graph_scalars: Optional[List[str]] = None
    y_graph_vector: Optional[str] = None
    atomic_numbers_col: str = "physicsml_atom_numbers"
    coordinates_col: str = "physicsml_coordinates"
    node_attrs_col: str = "physicsml_atom_features"
    edge_attrs_col: str = "physicsml_bond_features"
    node_idxs_col: str = "physicsml_atom_idxs"
    edge_idxs_col: str = "physicsml_bond_idxs"
    total_atomic_energy_col: str = "physicsml_total_atomic_energy"
    cut_off: float = 5.0
    pbc: Optional[Tuple[bool, bool, bool]] = None
    cell: Optional[List[List[float]]] = None
    self_interaction: bool = False
    use_scaled_positions: bool = False
    max_nbins: int = int(1e6)
    pre_batch: Optional[Literal["in_memory", "on_disk"]] = None
    pre_batch_in_memory: bool = False  # TODO: Deprecate
    train_batch_size: Optional[int] = None  # TODO: Deprecate
    validation_batch_size: Optional[int] = None  # TODO: Deprecate


@dataclass(config=ConfigDict)
class PhysicsMLModelConfig(LightningConfig):
    compute_forces: bool = False
    datamodule: PhysicsMLDataModuleConfig = field(
        default_factory=PhysicsMLDataModuleConfig,
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(name="Adam", config={"lr": 1e-3}),
    )
    scheduler: Optional[SchedulerConfig] = field(default_factory=SchedulerConfig)
