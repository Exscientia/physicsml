from typing import Any, Dict, Optional

import datasets
from molflux.modelzoo.models.lightning.datamodule import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from physicsml.lightning.config import PhysicsMLModelConfig
from physicsml.lightning.graph_datasets.graph_dataset import GraphDataset
from physicsml.lightning.pre_batching_in_memory import (
    construct_in_memory_pre_batched_dataloader,
)
from physicsml.lightning.pre_batching_on_disk import (
    construct_on_disk_pre_batched_dataloader,
)


class PhysicsMLDataModule(LightningDataModule):
    model_config: PhysicsMLModelConfig

    def __init__(
        self,
        model_config: PhysicsMLModelConfig,
        train_data: Optional[Dict[Optional[str], datasets.Dataset]] = None,
        validation_data: Optional[Dict[Optional[str], datasets.Dataset]] = None,
        test_data: Optional[Dict[Optional[str], datasets.Dataset]] = None,
        predict_data: Optional[datasets.Dataset] = None,
        **kwargs: Any,
    ):
        del kwargs
        super().__init__(
            model_config=model_config,
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            predict_data=predict_data,
        )

    def prepare_dataset(
        self,
        data: datasets.Dataset,
        split: str,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dataset:
        return GraphDataset(
            dataset=data,
            x_features=self.model_config.x_features,
            y_features=self.model_config.y_features,
            with_y_features=(split != "predict"),
            atomic_numbers_col=self.model_config.datamodule.atomic_numbers_col,
            node_attrs_col=self.model_config.datamodule.node_attrs_col,
            edge_attrs_col=self.model_config.datamodule.edge_attrs_col,
            node_idxs_col=self.model_config.datamodule.node_idxs_col,
            edge_idxs_col=self.model_config.datamodule.edge_idxs_col,
            coordinates_col=self.model_config.datamodule.coordinates_col,
            total_atomic_energy_col=self.model_config.datamodule.total_atomic_energy_col,
            y_node_scalars=self.model_config.datamodule.y_node_scalars,
            y_node_vector=self.model_config.datamodule.y_node_vector,
            y_edge_scalars=self.model_config.datamodule.y_edge_scalars,
            y_edge_vector=self.model_config.datamodule.y_edge_vector,
            y_graph_scalars=self.model_config.datamodule.y_graph_scalars,
            y_graph_vector=self.model_config.datamodule.y_graph_vector,
            num_elements=self.model_config.datamodule.num_elements,
            self_interaction=self.model_config.datamodule.self_interaction,
            pbc=self.model_config.datamodule.pbc,
            cell=self.model_config.datamodule.cell,
            cut_off=self.model_config.datamodule.cut_off,
        )

    def _get_one_train_dataloader(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
    ) -> DataLoader:
        if self.model_config.datamodule.pre_batch == "in_memory":
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
            )
            return construct_in_memory_pre_batched_dataloader(
                dataloader,
                shuffle=True,
            )
        elif self.model_config.datamodule.pre_batch == "on_disk":
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
            )
            return construct_on_disk_pre_batched_dataloader(
                dataloader,
                shuffle=True,
                name="train",
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
                persistent_workers=bool(self.model_config.datamodule.num_workers),
                drop_last=True,
            )

    def _get_one_eval_dataloader(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
    ) -> DataLoader:
        if self.model_config.datamodule.pre_batch == "in_memory":
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
            )
            return construct_in_memory_pre_batched_dataloader(
                dataloader,
                shuffle=False,
            )
        elif self.model_config.datamodule.pre_batch == "on_disk":
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
            )
            return construct_on_disk_pre_batched_dataloader(
                dataloader,
                shuffle=False,
                name="validation",
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
                persistent_workers=bool(self.model_config.datamodule.num_workers),
            )
