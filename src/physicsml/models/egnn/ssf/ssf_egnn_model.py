from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.egnn.ssf.default_configs import (
    SSFEGNNModelConfig,
)
from physicsml.models.egnn.ssf.ssf_egnn_module import PooledSSFEGNNModule


class SsfEGNNModel(PhysicsMLModelBase[SSFEGNNModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[SSFEGNNModelConfig]:
        return SSFEGNNModelConfig

    def _instantiate_module(self) -> Any:
        return PooledSSFEGNNModule(
            model_config=self.model_config,
        )
