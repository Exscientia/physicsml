from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model_uncertainty import PhysicsMLUncertaintyModelBase
from physicsml.models.mace.mean_var.default_configs import (
    MeanVarMACEModelConfig,
)
from physicsml.models.mace.mean_var.mean_var_mace_module import (
    PooledMeanVarMACEModule,
)


class MeanVarMACEModel(PhysicsMLUncertaintyModelBase[MeanVarMACEModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[MeanVarMACEModelConfig]:
        return MeanVarMACEModelConfig

    def _instantiate_module(self) -> Any:
        return PooledMeanVarMACEModule(
            model_config=self.model_config,
        )
