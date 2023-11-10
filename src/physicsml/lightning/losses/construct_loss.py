from typing import Any, Dict

from physicsml.lightning.losses import _AVAILABLE_LOSSES
from physicsml.lightning.losses.stock_losses import _STOCK_LOSSES


def construct_loss(loss_config: Dict, column_name: str) -> Any:
    if loss_config["name"] in _STOCK_LOSSES.keys():
        return _AVAILABLE_LOSSES["LossBase"](loss_config, column_name)
    else:
        return _AVAILABLE_LOSSES[loss_config["name"]](loss_config, column_name)
