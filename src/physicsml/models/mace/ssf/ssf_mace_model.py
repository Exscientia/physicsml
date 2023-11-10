from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.mace.ssf.default_configs import (
    SSFMACEModelConfig,
)
from physicsml.models.mace.ssf.ssf_mace_module import PooledSSFMACEModule


class SsfMACEModel(PhysicsMLModelBase[SSFMACEModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[SSFMACEModelConfig]:
        return SSFMACEModelConfig

    def _instantiate_module(self) -> Any:
        return PooledSSFMACEModule(
            model_config=self.model_config,
        )
