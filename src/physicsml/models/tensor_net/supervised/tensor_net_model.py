from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.tensor_net.supervised.default_configs import (
    TensorNetModelConfig,
)
from physicsml.models.tensor_net.supervised.tensor_net_module import (
    PooledTensorNetModule,
)


class TensorNetModel(PhysicsMLModelBase[TensorNetModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[TensorNetModelConfig]:
        return TensorNetModelConfig

    def _instantiate_module(self) -> Any:
        return PooledTensorNetModule(
            model_config=self.model_config,
        )
