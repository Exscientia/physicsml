from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.egnn.adapter.adapter_egnn_module import (
    PooledAdapterEGNNModule,
)
from physicsml.models.egnn.adapter.default_configs import (
    AdapterEGNNModelConfig,
)


class AdapterEGNNModel(PhysicsMLModelBase[AdapterEGNNModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[AdapterEGNNModelConfig]:
        return AdapterEGNNModelConfig

    def _instantiate_module(self) -> Any:
        return PooledAdapterEGNNModule(
            model_config=self.model_config,
        )
