from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model_uncertainty import PhysicsMLUncertaintyModelBase
from physicsml.models.allegro.mean_var.default_configs import (
    MeanVarAllegroModelConfig,
)
from physicsml.models.allegro.mean_var.mean_var_allegro_module import (
    PooledMeanVarAllegroModule,
)


class MeanVarAllegroModel(PhysicsMLUncertaintyModelBase[MeanVarAllegroModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[MeanVarAllegroModelConfig]:
        return MeanVarAllegroModelConfig

    def _instantiate_module(self) -> Any:
        return PooledMeanVarAllegroModule(
            model_config=self.model_config,
        )
