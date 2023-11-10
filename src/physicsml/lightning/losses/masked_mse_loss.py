from typing import Dict

import torch

from physicsml.lightning.losses import _AVAILABLE_LOSSES


class MaskedMSELoss(torch.nn.Module):
    def __init__(self, loss_config: Dict, column_name: str) -> None:
        super().__init__()

        self.weight = loss_config.get("weight", 1.0)
        self.column_name = column_name
        self.loss_func = torch.nn.MSELoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        ref: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mask = ~ref[self.column_name].isnan()
        masked_pred = pred[self.column_name][mask]
        masked_ref = ref[self.column_name][mask]
        loss: torch.Tensor = self.weight * self.loss_func(
            masked_pred,
            masked_ref,
        )

        return loss


_AVAILABLE_LOSSES["MaskedMSELoss"] = MaskedMSELoss
