import logging
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import molflux.datasets
import molflux.splits
import torch
from datasets import Dataset
from molflux.modelzoo.models.lightning.config import (
    CompileConfig,
    DataModuleConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TransferLearningConfigBase,
)
from molflux.modelzoo.models.lightning.model import LightningModelBase
from molflux.modelzoo.typing import PredictionResult

import lightning.pytorch as pl
from physicsml.lightning.datamodule import PhysicsMLDataModule
from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.utils import OptionalDependencyImportError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
    def _datamodule_builder(self) -> type[PhysicsMLDataModule]:
        return PhysicsMLDataModule

    @property
    @abstractmethod
    def _config_builder(self) -> type[_PhysicsMLModelConfigT]: ...

    @abstractmethod
    def _instantiate_module(self) -> PhysicsMLModuleBase: ...

    def _train_multi_data(
        self,
        train_data: dict[str | None, Dataset],
        validation_data: dict[str | None, Dataset] | None = None,
        datamodule_config: DataModuleConfig | dict[str, Any] | None = None,
        trainer_config: TrainerConfig | dict[str, Any] | None = None,
        optimizer_config: OptimizerConfig | dict[str, Any] | None = None,
        scheduler_config: SchedulerConfig | dict[str, Any] | None = None,
        transfer_learning_config: TransferLearningConfigBase
        | dict[str, Any]
        | None = None,
        compile_config: CompileConfig | dict[str, Any] | bool | None = None,
        ckpt_path: str | None = None,
        internal_validation_split: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if internal_validation_split:
            log.warning("Doing internal validation split")
            assert validation_data is None, RuntimeError(
                "Can't do internal validation split if validation_data provided",
            )
            strategy = molflux.splits.load_from_dict(
                {
                    "name": internal_validation_split["name"],
                    "config": internal_validation_split.get("config", {}),
                    "presets": internal_validation_split.get("presets", {}),
                },
            )
            split_datasets: dict[str, dict[Any, Any]] = {
                "train": {},
                "validation": {},
            }
            for k, v in train_data.items():
                split_data = next(
                    molflux.datasets.split_dataset(
                        v,
                        strategy=strategy,
                        groups_column=internal_validation_split.get(
                            "groups_column",
                            None,
                        ),
                        target_column=internal_validation_split.get(
                            "target_column",
                            None,
                        ),
                    ),
                )
                assert len(split_data["train"]) > 0, RuntimeError(
                    "Validation split resulted in 0 points in train split",
                )
                assert len(split_data["validation"]) > 0, RuntimeError(
                    "Validation split resulted in 0 points in validation split",
                )
                assert len(split_data["test"]) == 0, RuntimeError(
                    "Validation split resulted in non-zero points in test split",
                )
                log.warn(
                    f"{k} validation split: train ({len(split_data['train'])}), validation ({len(split_data['validation'])})",
                )
                split_datasets["train"][k] = split_data["train"]
                split_datasets["validation"][k] = split_data["validation"]
            super()._train_multi_data(
                train_data=split_datasets["train"],
                validation_data=split_datasets["validation"],
                datamodule_config=datamodule_config,
                trainer_config=trainer_config,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config,
                transfer_learning_config=transfer_learning_config,
                compile_config=compile_config,
                ckpt_path=ckpt_path,
                **kwargs,
            )
        else:
            super()._train_multi_data(
                train_data=train_data,
                validation_data=validation_data,
                datamodule_config=datamodule_config,
                trainer_config=trainer_config,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config,
                transfer_learning_config=transfer_learning_config,
                compile_config=compile_config,
                ckpt_path=ckpt_path,
                **kwargs,
            )

    def _predict_batched(
        self,
        data: Dataset,
        datamodule_config: DataModuleConfig | dict[str, Any] | None = None,
        trainer_config: TrainerConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        del kwargs

        with self.override_config(datamodule=datamodule_config, trainer=trainer_config):
            datamodule = self._instantiate_datamodule(predict_data=data)
            trainer_config_tmp = self.model_config.trainer.pass_to_trainer()
            trainer = pl.Trainer(
                accelerator=trainer_config_tmp["accelerator"],
                devices=trainer_config_tmp["devices"],
                strategy=trainer_config_tmp["strategy"],
                logger=False,
            )
            # Expect a list of Tensors, which may need to be overwritten for some Torch models
            batch_preds: list[dict[str, torch.Tensor]] = trainer.predict(  # type: ignore
                self.module,
                datamodule,
            )

        return batch_preds

    def _predict(
        self,
        data: Dataset,
        datamodule_config: DataModuleConfig | dict[str, Any] | None = None,
        trainer_config: TrainerConfig | dict[str, Any] | None = None,
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
            for display_name, y_feature in zip(
                display_names,
                self.y_features,
                strict=False,
            )
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
