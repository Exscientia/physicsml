import copy
from typing import Dict

import torch

from physicsml.lightning.losses import _AVAILABLE_LOSSES
from physicsml.lightning.losses.loss_base import LossBase


class SerialBCEWithLogitsLoss(LossBase):
    def __init__(self, loss_config: Dict, column_name: str) -> None:
        super().__init__(loss_config=loss_config, column_name=column_name)

        tmp_loss_config = copy.deepcopy(loss_config)

        if "weight" in tmp_loss_config.get("config", {}):
            tmp_loss_config["config"]["weight"] = torch.tensor(
                tmp_loss_config["config"]["weight"],
            )

        if "pos_weight" in tmp_loss_config.get("config", {}):
            tmp_loss_config["config"]["pos_weight"] = torch.tensor(
                tmp_loss_config["config"]["pos_weight"],
            )

        self.loss_func = torch.nn.BCEWithLogitsLoss(**tmp_loss_config.get("config", {}))

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        ref: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss: torch.Tensor = self.weight * self.loss_func(
            pred[self.column_name],
            ref[self.column_name],
        )

        return loss


_AVAILABLE_LOSSES["SerialBCEWithLogitsLoss"] = SerialBCEWithLogitsLoss
