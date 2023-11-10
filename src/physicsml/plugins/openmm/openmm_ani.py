import logging
from typing import Any, Dict, Optional

import torch

from physicsml.models.ani.supervised.default_configs import ANIModelConfig
from physicsml.plugins.openmm.openmm_base import OpenMMModuleBase

logger = logging.getLogger(__name__)


class OpenMMANI(OpenMMModuleBase):
    model_config: ANIModelConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.batch_dict = self.make_batch(self.datapoint)
        del self.model_config

    def make_batch(self, datapoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_dict = {}

        species = datapoint[self.model_config.datamodule.atomic_numbers_col].unsqueeze(
            0,
        )

        total_atomic_energy = datapoint.get(
            self.model_config.datamodule.total_atomic_energy_col,
            None,
        )

        batch_dict["species"] = species.type(torch.int64)

        if total_atomic_energy is not None:
            batch_dict["total_atomic_energy"] = total_atomic_energy.type(self.dtype)

        if self.pbc is not None:
            batch_dict["pbc"] = torch.tensor(self.pbc)

        if self.cell is not None:
            batch_dict["cell"] = torch.tensor(self.cell).type(self.dtype)

        return batch_dict

    def forward(
        self,
        positions: torch.Tensor,
        boxvectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_dict_clone = self.clone_batch_dict_to_device()

        # occasionally torchscript will squeeze tensors with shape (1, *) without warning.
        if batch_dict_clone["species"].dim() < 2:
            batch_dict_clone["species"] = batch_dict_clone["species"].unsqueeze(0)

        # scale positions
        positions = positions * self.position_scaling

        # truncate positions tensor if using mixed system
        if self.atom_idxs is not None:
            positions = positions[self.atom_idxs]

        batch_dict_clone["coordinates"] = positions.unsqueeze(0).to(self.which_device)

        # if box vectors are provided, override
        if boxvectors is not None:
            batch_dict_clone["cell"] = (
                boxvectors.type(positions.dtype) * self.position_scaling
            ).to(self.which_device)
            batch_dict_clone["pbc"] = torch.tensor([True, True, True]).to(
                self.which_device,
            )

        # do inference
        output: Dict[str, torch.Tensor] = self.module(batch_dict_clone)

        # get output and scaling
        y_out: torch.Tensor = output[self.y_output].squeeze()
        y_out = y_out * self.output_scaling

        return y_out
