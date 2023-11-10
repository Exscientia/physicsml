import os
import pickle
from typing import Any, List

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

from lightning.pytorch.utilities.rank_zero import rank_zero_only


class InMemoryBatchedPickledDataset(Dataset):
    def __init__(self, batched_dataset: List[Any]) -> None:
        super().__init__()
        self.batched_dataset = batched_dataset

    def __len__(self) -> int:
        return len(self.batched_dataset)

    def __getitem__(self, idx: int) -> Any:
        return pickle.loads(self.batched_dataset[idx])  # noqa: S301


def collate_fn(list_of_data: List[Any]) -> Any:
    assert len(list_of_data) == 1
    return list_of_data[0]


@rank_zero_only
def pre_batch_in_memory(dataloader: DataLoader) -> None:
    batches = []
    for b in tqdm(dataloader, desc="Pre-batching data in memory"):
        batches.append(pickle.dumps(b))

    torch.save(InMemoryBatchedPickledDataset(batches), "dataset.torch")


@rank_zero_only
def clean_up() -> None:
    os.remove("dataset.torch")


def construct_in_memory_pre_batched_dataloader(
    dataloader: DataLoader,
    shuffle: bool,
) -> DataLoader:
    pre_batch_in_memory(dataloader)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    batched_dataset = torch.load("dataset.torch")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    clean_up()

    return DataLoader(
        batched_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )
