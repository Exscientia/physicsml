import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import molflux.core as molflux_core
from datasets import Dataset
from datasets.utils import disable_progress_bar

from ase.calculators.calculator import Calculator
from physicsml.backends.backend_selector import atoms_or_file_to_bytes
from physicsml.lightning.model import PhysicsMLModelBase

if TYPE_CHECKING:
    from physicsml.lightning.module import PhysicsMLModuleBase

disable_progress_bar()
logger = logging.getLogger(__name__)


class PhysicsMLASECalculatorBase(Calculator):
    def __init__(
        self,
        physicsml_model: PhysicsMLModelBase,
        featurisation_metadata: Dict,
        y_output: Optional[str] = None,
        pbc: Optional[Tuple[bool, bool, bool]] = None,
        cell: Optional[List[List[float]]] = None,
        output_scaling: Optional[float] = None,
        position_scaling: Optional[float] = None,
        precision: str = "32",
        device: str = "cpu",
    ):
        super().__init__()

        self.implemented_properties: List[str] = ["energy", "forces"]

        self.model_config = physicsml_model.model_config
        self.model_config.datamodule.predict.batch_size = 1
        self.module: PhysicsMLModuleBase = physicsml_model.module
        self.module.to(device)
        self.module.eval()
        # set model precision
        if precision == "32":
            self.module.float()
        elif precision == "64":
            self.module.double()
        else:
            raise KeyError(f"Precision {precision} unknown. Use either '32' or '64'.")
        logger.warning(f"Model is of dtype {self.module.dtype}")

        self._instantiate_datamodule = physicsml_model._instantiate_datamodule
        self.featurisation_metadata = featurisation_metadata
        self.featurisation_metadata["config"][0]["column"] = "tmp_mol"

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

    def system_to_feated_dataset(
        self,
        atom_list: List,
        positions: List[List],
    ) -> Dataset:
        scaled_positions = [[xs * self.position_scaling for xs in x] for x in positions]
        self.featurisation_metadata["config"][0]["representations"][0]["config"][
            "backend"
        ] = self.featurisation_metadata["config"][0]["representations"][0][
            "config"
        ].get(
            "backend",
            "openeye",
        )
        backend = self.featurisation_metadata["config"][0]["representations"][0][
            "config"
        ]["backend"]
        system_bytes = atoms_or_file_to_bytes(backend)(
            atom_list=atom_list,
            coordinates=scaled_positions,
        )
        dataset = Dataset.from_dict({"tmp_mol": [system_bytes]})
        dataset_feated = molflux_core.featurise_dataset(
            dataset,
            self.featurisation_metadata,
        )

        return dataset_feated
