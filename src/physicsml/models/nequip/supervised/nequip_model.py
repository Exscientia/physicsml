from typing import Any

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.nequip.supervised.default_configs import (
    NequipModelConfig,
)
from physicsml.models.nequip.supervised.nequip_module import PooledNequipModule


class NequipModel(PhysicsMLModelBase[NequipModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> type[NequipModelConfig]:
        return NequipModelConfig

    def _instantiate_module(self) -> Any:
        return PooledNequipModule(
            model_config=self.model_config,
        )
