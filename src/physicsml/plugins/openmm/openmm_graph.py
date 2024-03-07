import logging
from typing import Any, Dict, Optional

import torch

from physicsml.lightning.config import PhysicsMLModelConfig
from physicsml.lightning.graph_datasets.neighbourhood_list_torch import (
    construct_edge_indices_and_attrs,
)
from physicsml.plugins.openmm.openmm_base import OpenMMModuleBase

logger = logging.getLogger(__name__)


class OpenMMGraph(OpenMMModuleBase):
    model_config: PhysicsMLModelConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Extract data from datapoint

        self.cut_off = self.model_config.datamodule.cut_off
        self.num_elements = self.model_config.datamodule.num_elements
        self.self_interaction = self.model_config.datamodule.self_interaction

        self.batch_dict = self.make_batch(self.datapoint)

        del self.model_config

    def make_batch(self, datapoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw_atomic_numbers = datapoint.get(
            self.model_config.datamodule.atomic_numbers_col,
            None,
        )
        node_attrs = datapoint.get(
            self.model_config.datamodule.node_attrs_col,
            None,
        )
        initial_edge_attrs = datapoint.get(
            self.model_config.datamodule.edge_attrs_col,
            None,
        )
        initial_edge_indices = datapoint.get(
            self.model_config.datamodule.edge_idxs_col,
            None,
        )
        total_atomic_energy = datapoint.get(
            self.model_config.datamodule.total_atomic_energy_col,
            None,
        )

        if raw_atomic_numbers is not None:
            atomic_numbers: Optional[torch.Tensor] = (
                torch.nn.functional.one_hot(
                    raw_atomic_numbers,
                    num_classes=self.model_config.datamodule.num_elements,
                )
                * 1.0
            )
        else:
            atomic_numbers = None

        if total_atomic_energy is not None:
            total_atomic_energy = total_atomic_energy.unsqueeze(0)

        if (node_attrs is not None) and (atomic_numbers is not None):
            node_attrs = torch.cat([atomic_numbers, node_attrs], dim=1) * 1.0
        elif atomic_numbers is not None:
            node_attrs = atomic_numbers * 1.0
        else:
            node_attrs = None

        # dataset will return list when in torch format with empty edge_attrs
        if (initial_edge_attrs is not None) and (len(initial_edge_attrs) == 0):
            initial_edge_attrs = torch.empty(0)

        # if no edge indices, add empty tensor in the same shape
        if (initial_edge_indices is not None) and (len(initial_edge_indices) == 0):
            initial_edge_indices = torch.empty(0, 2)

        # setting up the batch
        batch_dict = {}
        batch_dict["num_graphs"] = torch.tensor(1)
        if raw_atomic_numbers is not None:
            batch_dict["raw_atomic_numbers"] = raw_atomic_numbers
            batch_dict["num_nodes"] = torch.tensor(raw_atomic_numbers.shape[0])
            batch_dict["batch"] = torch.zeros(
                raw_atomic_numbers.shape[0],
                dtype=torch.int64,
            )
            batch_dict["ptr"] = torch.tensor(
                [0, raw_atomic_numbers.shape[0]],
                dtype=torch.int64,
            )
        if atomic_numbers is not None:
            batch_dict["atomic_numbers"] = atomic_numbers
        if total_atomic_energy is not None:
            batch_dict["total_atomic_energy"] = total_atomic_energy.type(self.dtype)
        if node_attrs is not None:
            batch_dict["node_attrs"] = node_attrs.type(self.dtype)
        if self.cell is not None:
            batch_dict["cell"] = torch.tensor(self.cell)

        if initial_edge_attrs is not None:
            self.initial_edge_attrs: Optional[torch.Tensor] = initial_edge_attrs.type(
                self.dtype,
            )
        else:
            self.initial_edge_attrs = None

        if initial_edge_indices is not None:
            self.initial_edge_indices: Optional[torch.Tensor] = initial_edge_indices
        else:
            self.initial_edge_indices = None

        return batch_dict

    def forward(
        self,
        positions: torch.Tensor,
        boxvectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_dict_clone = self.clone_batch_dict_to_device()

        # scale positions
        positions = positions * self.position_scaling

        # truncate positions tensor if using mixed system
        if self.atom_idxs is not None:
            positions = positions[self.atom_idxs]

        batch_dict_clone["coordinates"] = positions.to(self.which_device)

        # if box vectors are provided, override
        if boxvectors is not None:
            batch_dict_clone["cell"] = (
                boxvectors.type(positions.dtype) * self.position_scaling
            ).to(self.which_device)
        if "cell" in batch_dict_clone:
            cell = batch_dict_clone["cell"]
            pbc = (True, True, True)
        else:
            cell = None
            pbc = None

        edge_indices, edge_attrs, cell_shift_vector = construct_edge_indices_and_attrs(
            positions=batch_dict_clone["coordinates"],
            initial_edge_indices=self.initial_edge_indices,
            initial_edge_attrs=self.initial_edge_attrs,
            cell=cell,
            pbc=pbc,
            cutoff=self.cut_off,
            self_interaction=self.self_interaction,
        )
        if edge_attrs is not None:
            edge_attrs = edge_attrs * 1.0
        if edge_indices is not None:
            edge_indices = edge_indices.type(torch.int64)

        # add tensors to batch
        batch_dict_clone["edge_index"] = edge_indices
        if edge_attrs is not None:
            batch_dict_clone["edge_attrs"] = edge_attrs
        if "cell" in batch_dict_clone:
            batch_dict_clone["cell_shift_vector"] = cell_shift_vector

        # do inference
        output: Dict[str, torch.Tensor] = self.module(batch_dict_clone)

        # get outputs and scale
        y_out: torch.Tensor = output[self.y_output].squeeze()
        y_out = y_out * self.output_scaling

        return y_out
