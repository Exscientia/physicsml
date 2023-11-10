from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.allegro.supervised.allegro_module import PooledAllegroModule
from physicsml.models.allegro.supervised.default_configs import (
    AllegroModelConfig,
)


class AllegroModel(PhysicsMLModelBase[AllegroModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[AllegroModelConfig]:
        return AllegroModelConfig

    def _instantiate_module(self) -> Any:
        return PooledAllegroModule(
            model_config=self.model_config,
        )
