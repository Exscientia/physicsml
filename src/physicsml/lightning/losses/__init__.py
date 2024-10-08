from typing import Any

_AVAILABLE_LOSSES: dict[str, Any] = {}

from physicsml.lightning.losses.loss_base import *
from physicsml.lightning.losses.masked_mse_loss import *
from physicsml.lightning.losses.multitask_losses import *
from physicsml.lightning.losses.serial_bce_w_logits_loss import *
from physicsml.lightning.losses.weighted_mse_loss import *
