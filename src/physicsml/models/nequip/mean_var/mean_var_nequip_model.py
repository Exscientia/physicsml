from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model_uncertainty import PhysicsMLUncertaintyModelBase
from physicsml.models.nequip.mean_var.default_configs import (
    MeanVarNequipModelConfig,
)
from physicsml.models.nequip.mean_var.mean_var_nequip_module import (
    PooledMeanVarNequipModule,
)


class MeanVarNequipModel(PhysicsMLUncertaintyModelBase[MeanVarNequipModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[MeanVarNequipModelConfig]:
        return MeanVarNequipModelConfig

    def _instantiate_module(self) -> Any:
        return PooledMeanVarNequipModule(
            model_config=self.model_config,
        )
