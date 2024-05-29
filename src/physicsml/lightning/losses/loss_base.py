from typing import Dict

import torch

from physicsml.lightning.losses import _AVAILABLE_LOSSES
from physicsml.lightning.losses.stock_losses import _STOCK_LOSSES


class LossBase(torch.nn.Module):
    def __init__(self, loss_config: Dict, column_name: str) -> None:
        super().__init__()

        self.weight = loss_config.get("weight", 1.0)
        self.column_name = column_name
        self.loss_func = _STOCK_LOSSES[loss_config["name"]](
            **loss_config.get("config", {}),
        )

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        ref: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss_vec: torch.Tensor = self.loss_func(
            pred[self.column_name],
            ref[self.column_name],
        )
        loss = torch.Tensor([0])
        for i in range(len(self.weight)):
            for j in range(3):
                loss += loss_vec[i][j] * self.weight[i]

        return loss


_AVAILABLE_LOSSES["LossBase"] = LossBase
