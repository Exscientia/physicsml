from typing import Any

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.nequip.ssf.default_configs import (
    SSFNequipModelConfig,
)
from physicsml.models.nequip.ssf.ssf_nequip_module import PooledSSFNequipModule


class SsfNequipModel(PhysicsMLModelBase[SSFNequipModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> type[SSFNequipModelConfig]:
        return SSFNequipModelConfig

    def _instantiate_module(self) -> Any:
        return PooledSSFNequipModule(
            model_config=self.model_config,
        )
