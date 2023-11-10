import logging
from typing import Any, List, Optional, Tuple

import torch

import ase
from ase.calculators.calculator import all_changes
from physicsml.plugins.ase.calculator import PhysicsMLASECalculatorBase

logger = logging.getLogger(__name__)


class ANIASECalculator(PhysicsMLASECalculatorBase):
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

        batch_dict = {}
        for k, v in batch.items():
            batch_dict[k] = v.to(self.module.device)

        # Calculate energy and forces
        batch_dict["coordinates"].requires_grad = True

        output = self.module(batch_dict)
        energy = output["y_graph_scalars"] * self.output_scaling
        grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
        gradient = torch.autograd.grad(
            outputs=[energy],  # [1, 1]
            inputs=[batch_dict["coordinates"]],  # [1, n_nodes, 3]
            grad_outputs=grad_outputs,  # type: ignore
            allow_unused=True,
        )[
            0
        ]  # [n_nodes, 3]

        if gradient is None:
            raise RuntimeWarning("Gradient is None")
        forces = -1 * gradient.squeeze(0)

        self.results["energy"] = energy.detach().cpu().item()
        self.results["forces"] = forces.detach().cpu().numpy()

        # Return the energy and forces
        return self.results["energy"], self.results["forces"]
