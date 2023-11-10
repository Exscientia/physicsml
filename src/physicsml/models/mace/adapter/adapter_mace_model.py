from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.mace.adapter.adapter_mace_module import (
    PooledAdapterMACEModule,
)
from physicsml.models.mace.adapter.default_configs import (
    AdapterMACEModelConfig,
)


class AdapterMACEModel(PhysicsMLModelBase[AdapterMACEModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[AdapterMACEModelConfig]:
        return AdapterMACEModelConfig

    def _instantiate_module(self) -> Any:
        return PooledAdapterMACEModule(
            model_config=self.model_config,
        )
