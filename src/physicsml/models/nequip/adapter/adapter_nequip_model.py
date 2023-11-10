from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.nequip.adapter.adapter_nequip_module import (
    PooledAdapterNequipModule,
)
from physicsml.models.nequip.adapter.default_configs import (
    AdapterNequipModelConfig,
)


class AdapterNequipModel(PhysicsMLModelBase[AdapterNequipModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[AdapterNequipModelConfig]:
        return AdapterNequipModelConfig

    def _instantiate_module(self) -> Any:
        return PooledAdapterNequipModule(
            model_config=self.model_config,
        )
