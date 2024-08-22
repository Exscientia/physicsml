from typing import Any

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.egnn.multi_graph.default_configs import (
    MultiGraphEGNNModelConfig,
)
from physicsml.models.egnn.multi_graph.multi_graph_datamodule import (
    MultiGraphDataModule,
)
from physicsml.models.egnn.multi_graph.multi_graph_egnn_module import (
    PooledMultiGraphEGNNModule,
)


class MultiGraphEGNNModel(PhysicsMLModelBase[MultiGraphEGNNModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> type[MultiGraphEGNNModelConfig]:
        return MultiGraphEGNNModelConfig

    def _instantiate_module(self) -> Any:
        return PooledMultiGraphEGNNModule(
            model_config=self.model_config,
        )

    @property
    def _datamodule_builder(self) -> type[MultiGraphDataModule]:  # type: ignore
        return MultiGraphDataModule
