from typing import Optional

import torch


class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: Optional[float], shift: Optional[float]) -> None:
        super().__init__()

        if scale is not None:
            self.register_buffer("scale", torch.tensor(scale))
        else:
            self.register_buffer("scale", torch.tensor(1.0))

        if shift is not None:
            self.register_buffer("shift", torch.tensor(shift))
        else:
            self.register_buffer("shift", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift
