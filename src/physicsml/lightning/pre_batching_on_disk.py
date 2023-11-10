from pathlib import Path
from typing import Any, List

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

from lightning.pytorch.utilities.rank_zero import rank_zero_only


class OnDiskBatchedDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = Path(path)
        self.length = len(
            [None for p in self.path.iterdir() if str(p).endswith(".torch")],
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Any:
        batch = torch.load(self.path / f"batch_{idx}.torch")

        return batch


def collate_fn(list_of_data: List[Any]) -> Any:
    assert len(list_of_data) == 1
    return list_of_data[0]


@rank_zero_only
def pre_batch_on_disk(dataloader: DataLoader, path: str) -> int:
    posix_path = Path(path)
    posix_path.mkdir()

    for idx, b in tqdm(
        enumerate(dataloader),
        desc="Pre-batching data on disk",
        total=len(dataloader),
    ):
        torch.save(b, posix_path / f"batch_{idx}.torch")

    return idx + 1  # type: ignore


def construct_on_disk_pre_batched_dataloader(
    dataloader: DataLoader,
    shuffle: bool,
    name: str,
) -> DataLoader:
    pre_batch_on_disk(dataloader, f"pre_batched_{name}_dataset")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    batched_dataset = OnDiskBatchedDataset(
        path=f"pre_batched_{name}_dataset",
    )

    return DataLoader(
        batched_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )
