from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model_uncertainty import PhysicsMLUncertaintyModelBase
from physicsml.models.egnn.mean_var.default_configs import (
    MeanVarEGNNModelConfig,
)
from physicsml.models.egnn.mean_var.mean_var_egnn_module import (
    PooledMeanVarEGNNModule,
)


class MeanVarEGNNModel(PhysicsMLUncertaintyModelBase[MeanVarEGNNModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[MeanVarEGNNModelConfig]:
        return MeanVarEGNNModelConfig

    def _instantiate_module(self) -> Any:
        return PooledMeanVarEGNNModule(
            model_config=self.model_config,
        )
