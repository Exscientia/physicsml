from typing import Dict

import torch

from physicsml.lightning.losses import _AVAILABLE_LOSSES


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, loss_config: Dict, column_name: str) -> None:
        super().__init__()
        self.weight = loss_config.get("weight", 1.0)
        self.column_name = column_name

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        ref: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        num_atoms = ref["ptr"][1:] - ref["ptr"][:-1]  # [n_graphs,]

        loss: torch.Tensor = self.weight * torch.mean(
            torch.square((ref[self.column_name] - pred[self.column_name]) / num_atoms),
        )
        return loss


_AVAILABLE_LOSSES["WeightedMSELoss"] = WeightedMSELoss
