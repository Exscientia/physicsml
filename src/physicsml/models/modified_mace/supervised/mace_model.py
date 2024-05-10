from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.modified_mace.supervised.default_configs import (
    ModifiedMACEModelConfig,
)
from physicsml.models.modified_mace.supervised.mace_module import PooledMACEModule


class ModifiedMACEModel(PhysicsMLModelBase[ModifiedMACEModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[ModifiedMACEModelConfig]:
        return ModifiedMACEModelConfig

    def _instantiate_module(self) -> Any:
        return PooledMACEModule(
            model_config=self.model_config,
        )
