import logging
from typing import Any, List, Optional, Tuple

import ase
from ase.calculators.calculator import all_changes
from physicsml.plugins.ase.calculator import PhysicsMLASECalculatorBase

logger = logging.getLogger(__name__)


class GraphASECalculator(PhysicsMLASECalculatorBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def calculate(
        self,
        atoms: Optional[ase.Atoms] = None,
        properties: Optional[List[str]] = None,
        system_changes: Optional[List[str]] = all_changes,
    ) -> Tuple:
        super().calculate(atoms, properties, system_changes)

        # create system
        atom_list = self.atoms.get_atomic_numbers().tolist()
        positions = self.atoms.get_positions().tolist()
        dataset_feated = self.system_to_feated_dataset(
            atom_list=atom_list,
            positions=positions,
        )

        batch = next(
            iter(
                self._instantiate_datamodule(
                    predict_data=dataset_feated,
                ).predict_dataloader(),
            ),
        )

        batch_dict = self.module.graph_batch_to_batch_dict(batch.to(self.module.device))

        # Calculate energy and forces
        batch_dict["coordinates"].requires_grad = True

        output = self.module(batch_dict)
        energy = output["y_graph_scalars"] * self.output_scaling
        forces = self.module.compute_forces_by_gradient(
            energy=output["y_graph_scalars"],
            coordinates=batch_dict["coordinates"],
        )

        self.results["energy"] = energy.detach().cpu().item()
        self.results["forces"] = forces.detach().cpu().numpy()

        # Return the energy and forces
        return self.results["energy"], self.results["forces"]
