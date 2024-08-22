from typing import Any

import torch
from torch_geometric.loader.dataloader import Collater

from physicsml.models.egnn.multi_graph.multi_graph_dataset import (
    MultiGraphDataset,
)


class MultiGraphCollate:
    def __init__(self, dataset: MultiGraphDataset) -> None:
        self.collator_dict = {}

        for k, v in dataset.graph_datasets.items():
            self.collator_dict[k] = Collater(v, None, None)

    def __call__(self, batch: list[Any]) -> dict[str, Any]:
        output = {}
        batch_dict = {k: [dic[k] for dic in batch] for k in batch[0]}

        for k, v in batch_dict.items():
            output[k] = self.collator_dict[k](v)

        return output


class MultiGraphDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: MultiGraphDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs: Any,
    ) -> None:
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=MultiGraphCollate(dataset),
            **kwargs,
        )
