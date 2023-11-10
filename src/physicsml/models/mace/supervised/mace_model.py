from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.mace.supervised.default_configs import (
    MACEModelConfig,
)
from physicsml.models.mace.supervised.mace_module import PooledMACEModule


class MACEModel(PhysicsMLModelBase[MACEModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[MACEModelConfig]:
        return MACEModelConfig

    def _instantiate_module(self) -> Any:
        return PooledMACEModule(
            model_config=self.model_config,
        )
