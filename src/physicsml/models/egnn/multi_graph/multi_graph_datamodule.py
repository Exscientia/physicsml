from typing import Any

import datasets
from molflux.modelzoo.models.lightning.datamodule import LightningDataModule
from torch.utils.data import Dataset

from physicsml.lightning.pre_batching_in_memory import (
    construct_in_memory_pre_batched_dataloader,
)
from physicsml.lightning.pre_batching_on_disk import (
    construct_on_disk_pre_batched_dataloader,
)
from physicsml.models.egnn.multi_graph.default_configs import (
    MultiGraphEGNNModelConfig,
)
from physicsml.models.egnn.multi_graph.multi_graph_dataloader import (
    MultiGraphDataLoader,
)
from physicsml.models.egnn.multi_graph.multi_graph_dataset import (
    MultiGraphDataset,
)


class MultiGraphDataModule(LightningDataModule):
    model_config: MultiGraphEGNNModelConfig

    def __init__(
        self,
        model_config: MultiGraphEGNNModelConfig,
        train_data: dict[str | None, datasets.Dataset] | None = None,
        validation_data: dict[str | None, datasets.Dataset] | None = None,
        test_data: dict[str | None, datasets.Dataset] | None = None,
        predict_data: datasets.Dataset | None = None,
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
        name: str | None = None,
        **kwargs: Any,
    ) -> Dataset:
        return MultiGraphDataset(
            dataset=data,
            x_features=self.model_config.x_features,
            y_features=self.model_config.y_features,
            train_features=self.model_config.train_features,  # type: ignore
            with_y_features=(split != "predict"),
            graph_names=self.model_config.graph_names,
            dict_atomic_numbers_col={
                graph_name: f"{graph_name}::{self.model_config.datamodule.atomic_numbers_col}"
                for graph_name in self.model_config.graph_names
            },
            dict_node_attrs_col={
                graph_name: f"{graph_name}::{self.model_config.datamodule.node_attrs_col}"
                for graph_name in self.model_config.graph_names
            },
            dict_edge_attrs_col={
                graph_name: f"{graph_name}::{self.model_config.datamodule.edge_attrs_col}"
                for graph_name in self.model_config.graph_names
            },
            dict_node_idxs_col={
                graph_name: f"{graph_name}::{self.model_config.datamodule.node_idxs_col}"
                for graph_name in self.model_config.graph_names
            },
            dict_edge_idxs_col={
                graph_name: f"{graph_name}::{self.model_config.datamodule.edge_idxs_col}"
                for graph_name in self.model_config.graph_names
            },
            dict_coordinates_col={
                graph_name: f"{graph_name}::{self.model_config.datamodule.coordinates_col}"
                for graph_name in self.model_config.graph_names
            },
            y_graph_scalars=self.model_config.datamodule.y_graph_scalars,
            num_elements=self.model_config.datamodule.num_elements,
            self_interaction=self.model_config.datamodule.self_interaction,
            pbc=self.model_config.datamodule.pbc,
            cell=self.model_config.datamodule.cell,
            cut_off=self.model_config.datamodule.cut_off,
        )

    def _get_one_train_dataloader(
        self,
        dataset: MultiGraphDataset,
        batch_size: int,
    ) -> MultiGraphDataLoader:
        if self.model_config.datamodule.pre_batch == "in_memory":
            dataloader = MultiGraphDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
            )
            return construct_in_memory_pre_batched_dataloader(  # type: ignore
                dataloader,
                shuffle=True,
            )
        elif self.model_config.datamodule.pre_batch == "on_disk":
            dataloader = MultiGraphDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
            )
            return construct_on_disk_pre_batched_dataloader(  # type: ignore
                dataloader,
                shuffle=True,
                name="train",
            )
        else:
            return MultiGraphDataLoader(
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
        dataset: MultiGraphDataset,
        batch_size: int,
    ) -> MultiGraphDataLoader:
        if self.model_config.datamodule.pre_batch == "in_memory":
            dataloader = MultiGraphDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
            )
            return construct_in_memory_pre_batched_dataloader(  # type: ignore
                dataloader,
                shuffle=False,
            )
        elif self.model_config.datamodule.pre_batch == "on_disk":
            dataloader = MultiGraphDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
            )
            return construct_on_disk_pre_batched_dataloader(  # type: ignore
                dataloader,
                shuffle=False,
                name="validation",
            )
        else:
            return MultiGraphDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=int(self.model_config.datamodule.num_workers or 0),
                persistent_workers=bool(self.model_config.datamodule.num_workers),
            )
