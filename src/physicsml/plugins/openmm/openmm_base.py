import logging
from abc import abstractmethod
from typing import Any

import datasets
import molflux.core as molflux_core
import torch

from physicsml.backends.backend_selector import atoms_or_file_to_bytes
from physicsml.lightning.module import PhysicsMLModuleBase

logger = logging.getLogger(__name__)
datasets.disable_progress_bar()
datasets.disable_caching()


class OpenMMModuleBase(PhysicsMLModuleBase):
    def __init__(
        self,
        physicsml_model: Any,
        featurisation_metadata: dict,
        atom_list: list[int] | None = None,
        system_path: str | None = None,
        atom_idxs: list[int] | None = None,
        total_charge: int | None = None,
        y_output: str | None = None,
        pbc: tuple[bool, bool, bool] | None = None,
        cell: list[list[float]] | None = None,
        output_scaling: float | None = None,
        position_scaling: float | None = None,
        precision: str = "32",
        device: str = "cpu",
    ) -> None:
        super().__init__(physicsml_model.model_config)

        # get device
        self.which_device = device

        # copy model, delete losses (not torchscript friendly),
        # and set model to eval and send to device
        self.module = physicsml_model.module
        del self.module.model_config
        if hasattr(self.module, "losses"):
            del self.module.losses
        self.to(self.which_device)
        self.eval()
        # set model precision
        if precision == "32":
            self.float()
            self.module.float()
        elif precision == "64":
            self.double()
            self.module.double()
        else:
            raise KeyError(f"Precision {precision} unknown. Use either '32' or '64'.")
        self.model_dtype = self.dtype
        logger.warning(f"Model is of dtype {self.dtype}")

        # get scaling values for output and input
        self.output_scaling = output_scaling or 1.0
        self.position_scaling = position_scaling or 1.0

        # will output scalars (assumed to be energy) unless otherwise specified)
        self.y_output = y_output or "y_graph_scalars"
        logger.warning(
            f"Model will output {getattr(self.model_config.datamodule, self.y_output, None)}",
        )

        # specify pbcs and cell
        self.pbc = pbc
        self.cell = cell
        assert ((self.pbc is not None) and (self.cell is not None)) or (
            (self.pbc is None) and (self.cell is None)
        ), ValueError("Must specify both cel and pbc or neither.")
        if self.cell is not None:
            self.cell = [
                [float(el) * self.position_scaling for el in vector]
                for vector in self.cell
            ]

        # check if using tructated atoms list (for mixed systems) and specify atom idxs tensor
        if atom_idxs is not None:
            self.atom_idxs: torch.Tensor | None = torch.tensor(
                atom_idxs,
                dtype=torch.int64,
            )
        else:
            self.atom_idxs = None

        # total molecular charge
        # set default to None
        self.total_charge: int | None = None

        if total_charge is not None:
            self.total_charge = total_charge

        self.featurisation_metadata = featurisation_metadata
        self.featurisation_metadata["config"][0]["column"] = "tmp_mol"
        self.featurisation_metadata["config"][0]["representations"][0]["config"][
            "backend"
        ] = self.featurisation_metadata["config"][0]["representations"][0][
            "config"
        ].get(
            "backend",
            "openeye",
        )
        self.backend = self.featurisation_metadata["config"][0]["representations"][0][
            "config"
        ]["backend"]

        system_bytes = atoms_or_file_to_bytes(self.backend)(
            atom_list=atom_list,
            system_path=system_path,
        )

        # featurise system
        dataset = datasets.Dataset.from_dict({"tmp_mol": [system_bytes]})
        dataset_feated = molflux_core.featurise_dataset(
            dataset,
            self.featurisation_metadata,
        )
        dataset_feated.set_format("torch", self.model_config.x_features)
        self.datapoint = dataset_feated[0]
        self.batch_dict: dict[str, torch.Tensor]

    @abstractmethod
    def make_batch(
        self,
        datapoint: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]: ...

    def clone_batch_dict_to_device(self) -> dict[str, torch.Tensor]:
        clone = {}

        for k, v in self.batch_dict.items():
            clone[k] = v.clone().to(self.which_device)

        # add total_charge to the clone if it exists
        if self.total_charge is not None:
            clone["graph_attrs"] = torch.as_tensor(
                [[self.total_charge]],
                dtype=torch.float,
            ).to(self.which_device)

        return clone

    def compute_loss(self, input: Any, target: Any) -> dict[str, torch.Tensor]:
        return {"loss": torch.empty(0)}
