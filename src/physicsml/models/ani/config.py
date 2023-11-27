import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

from molflux.modelzoo.models.lightning.config import (
    DataModuleConfig,
)
from pydantic import validator
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


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

    @validator("pre_batch_in_memory")
    def deprecated_pre_batch_in_memory(
        cls,
        pre_batch_in_memory: bool,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        if pre_batch_in_memory and (values["pre_batch"] is None):
            logger.warn(
                "The 'pre_batch_in_memory' kwarg is deprecated. Use 'pre_batch': 'in_memory'.",
            )
            values["pre_batch"] = "in_memory"

        return pre_batch_in_memory

    @validator("train_batch_size")
    def deprecated_train_batch_size(
        cls,
        train_batch_size: Optional[int],
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Optional[int]:
        if train_batch_size:
            logger.warn(
                "The 'train_batch_size' kwarg is deprecated. Use 'train': {'batch_size': batch_size}.",
            )
            values["train"].batch_size = train_batch_size

        return train_batch_size

    @validator("validation_batch_size")
    def deprecated_validation_batch_size(
        cls,
        validation_batch_size: Optional[int],
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Optional[int]:
        if validation_batch_size:
            logger.warn(
                "The 'validation_batch_size' kwarg is deprecated. Use 'validation': {'batch_size': batch_size}.",
            )
            values["validation"].batch_size = validation_batch_size

        return validation_batch_size
