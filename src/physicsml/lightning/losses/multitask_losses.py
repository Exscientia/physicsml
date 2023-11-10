from typing import Dict

import torch

from physicsml.lightning.losses import _AVAILABLE_LOSSES
from physicsml.lightning.losses.loss_base import LossBase


class MultiTaskLoss(torch.nn.Module):
    def __init__(self, loss_config: Dict, column_name: str) -> None:
        super().__init__()
        self.column_name = column_name

        losses_configs = loss_config["config"]["losses_configs"]
        self.losses = []
        for loss_config in losses_configs:
            self.losses.append(LossBase(loss_config, column_name))

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        ref: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pred_tensor = pred[self.column_name]
        ref_tensor = ref[self.column_name]

        loss = []
        num_tasks = ref_tensor.shape[-1]
        for i in range(num_tasks):
            ref_tensor_i = ref_tensor[:, i]
            ref_tensor_i_mask = ~ref_tensor_i.isnan()
            masked_ref_tensor_i = ref_tensor_i[ref_tensor_i_mask]
            ref_dict = {self.column_name: masked_ref_tensor_i}

            pred_tensor_i = pred_tensor[:, i]
            masked_pred_tensor_i = pred_tensor_i[ref_tensor_i_mask]
            pred_dict = {self.column_name: masked_pred_tensor_i}

            if ref_tensor_i_mask.sum() > 0:
                loss_i = self.losses[i](pred_dict, ref_dict)
                loss.append(loss_i / num_tasks)

        total_loss: torch.Tensor = torch.stack(loss).sum()

        return total_loss


_AVAILABLE_LOSSES["MultiTaskLoss"] = MultiTaskLoss
