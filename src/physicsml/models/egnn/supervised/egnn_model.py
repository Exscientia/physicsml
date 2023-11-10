from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.egnn.supervised.default_configs import (
    EGNNModelConfig,
)
from physicsml.models.egnn.supervised.egnn_module import PooledEGNNModule


class EGNNModel(PhysicsMLModelBase[EGNNModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[EGNNModelConfig]:
        return EGNNModelConfig

    def _instantiate_module(self) -> Any:
        return PooledEGNNModule(
            model_config=self.model_config,
        )
