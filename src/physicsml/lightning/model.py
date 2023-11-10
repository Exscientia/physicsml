from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Type,
    TypeVar,
    Union,
)

import torch
from datasets import Dataset
from molflux.modelzoo.models.lightning.config import DataModuleConfig, TrainerConfig
from molflux.modelzoo.models.lightning.model import LightningModelBase
from molflux.modelzoo.typing import PredictionResult

import lightning.pytorch as pl
from physicsml.lightning.datamodule import PhysicsMLDataModule
from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.utils import OptionalDependencyImportError

if TYPE_CHECKING:
    from physicsml.lightning.config import PhysicsMLModelConfig

_PhysicsMLModelConfigT = TypeVar(
    "_PhysicsMLModelConfigT",
    bound="PhysicsMLModelConfig",
)


class PhysicsMLModelBase(
    LightningModelBase[_PhysicsMLModelConfigT],
):
    """ABC for all Huggingface PhysicsML models"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def _datamodule_builder(self) -> Type[PhysicsMLDataModule]:
        return PhysicsMLDataModule

    @property
    @abstractmethod
    def _config_builder(self) -> Type[_PhysicsMLModelConfigT]:
        ...

    @abstractmethod
    def _instantiate_module(self) -> PhysicsMLModuleBase:
        ...

    def _predict_batched(
        self,
        data: Dataset,
        datamodule_config: Union[DataModuleConfig, Dict[str, Any], None] = None,
        trainer_config: Union[TrainerConfig, Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> List[Any]:
        del kwargs

        with self.override_config(datamodule=datamodule_config, trainer=trainer_config):
            datamodule = self._instantiate_datamodule(predict_data=data)
            trainer_config = self.model_config.trainer.pass_to_trainer()
            trainer = pl.Trainer(
                accelerator=trainer_config["accelerator"],
                devices=trainer_config["devices"],
                strategy=trainer_config["strategy"],
                logger=False,
            )
            # Expect a list of Tensors, which may need to be overwritten for some Torch models
            batch_preds: List[Dict[str, torch.Tensor]] = trainer.predict(  # type: ignore
                self.module,
                datamodule,
            )

        return batch_preds

    def _predict(
        self,
        data: Dataset,
        datamodule_config: Union[DataModuleConfig, Dict[str, Any], None] = None,
        trainer_config: Union[TrainerConfig, Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> PredictionResult:
        """method for predicting"""
        del kwargs

        display_names = self._predict_display_names
        # if data is empty
        if not len(data):
            return {display_name: [] for display_name in display_names}

        batch_preds = self._predict_batched(
            data=data,
            datamodule_config=datamodule_config,
            trainer_config=trainer_config,
        )
        batched_preds_dict = {
            k: [dic[k] for dic in batch_preds] for k in batch_preds[0]
        }

        output = {}
        for key, cols in [
            ("y_node_scalars", self.model_config.datamodule.y_node_scalars),
            ("y_edge_scalars", self.model_config.datamodule.y_edge_scalars),
            ("y_graph_scalars", self.model_config.datamodule.y_graph_scalars),
        ]:
            if cols is not None:
                catted_value = torch.cat(batched_preds_dict[key], dim=0)
                for idx, col in enumerate(cols):
                    output[col] = catted_value[:, idx].tolist()

        for key, vec_col in [
            ("y_node_vector", self.model_config.datamodule.y_node_vector),
            ("y_edge_vector", self.model_config.datamodule.y_edge_vector),
            ("y_graph_vector", self.model_config.datamodule.y_graph_vector),
        ]:
            if vec_col is not None:
                output[vec_col] = torch.cat(batched_preds_dict[key], dim=0).tolist()

        return {
            display_name: output[y_feature]
            for display_name, y_feature in zip(display_names, self.y_features)
        }

    def to_openmm(self, **kwargs: Any) -> Any:
        from physicsml.plugins.openmm.openmm_graph import OpenMMGraph

        return OpenMMGraph(**kwargs)

    def to_ase(self, **kwargs: Any) -> Any:
        try:
            from physicsml.plugins.ase.ase_graph import GraphASECalculator
        except ImportError as err:
            raise OptionalDependencyImportError("ASE", "ase") from err
        else:
            return GraphASECalculator(**kwargs)
