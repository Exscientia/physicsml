import gc
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from molflux.modelzoo.models.lightning.module import (
    SingleBatchStepOutput,
)
from torchani import ANIModel
from torchani.aev import AEVComputer

from physicsml.lightning.losses.construct_loss import construct_loss
from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.models.ani.ani_1_2_defaults import (
    ani_1_2_aev_configs,
    ani_1_2_net_sizes_dict,
)
from physicsml.models.ani.mean_var.default_configs import MeanVarANIModelConfig


def atomic_nets_to_module(net_sizes: Dict[str, Any]) -> Dict[str, torch.nn.Sequential]:
    atomic_nets = OrderedDict()
    for a in net_sizes.keys():
        layers = net_sizes[a]
        modules: List[Any] = []
        for i in range(len(layers) - 1):
            modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
            modules.append(torch.nn.CELU(alpha=0.1))
        modules.append(torch.nn.Linear(layers[-1], 1))
        atomic_nets[a] = torch.nn.Sequential(*modules)

    return atomic_nets


class PooledMeanVarANIModule(PhysicsMLModuleBase):
    """
    Class for pooled ani model
    """

    model_config: MeanVarANIModelConfig

    def __init__(
        self,
        model_config: MeanVarANIModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config=model_config)

        self.losses = self.configure_losses()

        self.compute_forces = self.model_config.compute_forces
        self.scaling_mean = model_config.scaling_mean
        self.scaling_std = model_config.scaling_std

        self.net_sizes = ani_1_2_net_sizes_dict[model_config.which_ani]
        self.aev_config = ani_1_2_aev_configs[model_config.which_ani]

        self.animodel = ANIModel(atomic_nets_to_module(self.net_sizes))
        self.animodel_std = ANIModel(atomic_nets_to_module(self.net_sizes))

        self.aev_computer = AEVComputer(**self.aev_config)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "cell" in input:
            cell: Optional[torch.Tensor] = input["cell"]
        else:
            cell = None

        if "pbc" in input:
            pbc: Optional[torch.Tensor] = input["pbc"]
        else:
            pbc = None

        aev_output = self.aev_computer(
            (input["species"], input["coordinates"]),
            cell=cell,
            pbc=pbc,
        )

        animodel_output = self.animodel(aev_output, cell=cell, pbc=pbc)
        animodel_std_output = self.animodel_std(aev_output, cell=cell, pbc=pbc)

        energies = animodel_output.energies
        energies = energies * self.scaling_std + self.scaling_mean

        stds = torch.abs(animodel_std_output.energies) / (input["species"] != -1).sum(
            -1,
        )
        stds = stds * self.scaling_std

        if "total_atomic_energy" in input:
            energies = energies + input["total_atomic_energy"]

        energies = energies.unsqueeze(-1)
        stds = stds.unsqueeze(-1)

        output = {}

        output["y_graph_scalars"] = energies
        output["y_graph_scalars::std"] = stds

        return output

    def compute_loss(self, input: Any, target: Any) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor = torch.zeros(1, device=self.device)
        for y_key, loss in self.losses.items():
            if y_key == "y_graph_scalars":
                key_loss = loss(
                    input["y_graph_scalars"],
                    target["y_graph_scalars"],
                    input["y_graph_scalars::std"] ** 2,
                )
                loss_dict[y_key] = key_loss
                total_loss += key_loss
            elif target.get(y_key, None) is not None:
                key_loss = loss(input, target)
                loss_dict[y_key] = key_loss
                total_loss += key_loss

        loss_dict["loss"] = total_loss
        return loss_dict

    def configure_losses(self) -> Any:
        losses: Dict[str, Optional[Any]] = {}
        losses["y_graph_scalars"] = torch.nn.GaussianNLLLoss(eps=0.01)

        if self.model_config.y_node_vector_loss_config is not None:
            losses["y_node_vector"] = construct_loss(
                loss_config=self.model_config.y_node_vector_loss_config,
                column_name="y_node_vector",
            )

        return losses

    def _training_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: Optional[str],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        """fit method for doing a training step"""

        for k, v in single_source_batch.items():
            single_source_batch[k] = v.to(self.device)

        if self.compute_forces:
            single_source_batch["coordinates"].requires_grad = True
            output = self(single_source_batch)
            forces = self.compute_forces_by_gradient(
                energy=output["y_graph_scalars"],
                coordinates=single_source_batch["coordinates"],
            )
            if output.get("y_node_vector", None) is not None:
                raise RuntimeError(
                    "Cannot compute forces is model already outputs y_node_vector.",
                )

            flat_forces = []
            for idx, mol in enumerate(single_source_batch["species"]):
                num_atoms = (mol != -1).sum()
                flat_forces.append(forces[idx, :num_atoms, :])

            output["y_node_vector"] = torch.cat(flat_forces, dim=0)
        else:
            output = self(single_source_batch)

        loss_dict = self.compute_loss(output, single_source_batch)

        return loss_dict["loss"], loss_dict, single_source_batch["species"].shape[0]

    def _validation_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: Optional[str],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        """fit method for doing a validation step"""

        for k, v in single_source_batch.items():
            single_source_batch[k] = v.to(self.device)

        if self.compute_forces:
            with torch.enable_grad():
                single_source_batch["coordinates"].requires_grad = True
                output = self(single_source_batch)
                forces = self.compute_forces_by_gradient(
                    energy=output["y_graph_scalars"],
                    coordinates=single_source_batch["coordinates"],
                )
                if output.get("y_node_vector", None) is not None:
                    raise RuntimeError(
                        "Cannot compute forces is model already outputs y_node_vector.",
                    )

                flat_forces = []
                for idx, mol in enumerate(single_source_batch["species"]):
                    num_atoms = (mol != -1).sum()
                    flat_forces.append(forces[idx, :num_atoms, :])

                output["y_node_vector"] = torch.cat(flat_forces, dim=0)
        else:
            output = self(single_source_batch)

        loss_dict = self.compute_loss(output, single_source_batch)

        return loss_dict["loss"], loss_dict, single_source_batch["species"].shape[0]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """fit method for doing a predict step"""

        if self.compute_forces:
            with torch.inference_mode(False):
                batch_clone = {}
                for k, v in batch.items():
                    batch_clone[k] = v.clone().to(self.device)
                batch_clone["coordinates"].requires_grad = True
                output = self(batch_clone)
                forces = self.compute_forces_by_gradient(
                    energy=output["y_graph_scalars"],
                    coordinates=batch_clone["coordinates"],
                )
                if output.get("y_node_vector", None) is not None:
                    raise RuntimeError(
                        "Cannot compute forces is model already outputs y_node_vector.",
                    )

                flat_forces = []
                for idx, mol in enumerate(batch_clone["species"]):
                    num_atoms = (mol != -1).sum()
                    flat_forces.append(forces[idx, :num_atoms, :])

                output["y_node_vector"] = torch.cat(flat_forces, dim=0)
        else:
            batch_clone = {}
            for k, v in batch.items():
                batch_clone[k] = v.clone().to(self.device)
            output = self(batch_clone)

        detached_output: Any
        detached_output = {}
        for k, v in output.items():
            detached_output[k] = v.detach()

        del output
        del batch
        del batch_clone
        gc.collect()

        return detached_output
