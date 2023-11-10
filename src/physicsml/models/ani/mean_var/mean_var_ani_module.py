import gc
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from molflux.modelzoo.models.lightning.module import (
    LightningModuleBase,
    SingleBatchStepOutput,
)
from torchani import ANIModel
from torchani.aev import AEVComputer

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


class PooledMeanVarANIModule(LightningModuleBase):
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

    def compute_loss(self, input: Any, target: Any) -> torch.Tensor:
        total_loss: torch.Tensor = torch.zeros(1, device=self.device)
        for y_key, loss in self.losses.items():
            if y_key == "y_graph_scalars":
                total_loss += loss(
                    input["y_graph_scalars"],
                    target["y_graph_scalars"],
                    input["y_graph_scalars::std"] ** 2,
                )
            elif target.get(y_key, None) is not None:
                total_loss += loss(input, target)

        return total_loss

    def configure_losses(self) -> Any:
        losses: Dict[str, Optional[Any]] = {}
        losses["y_graph_scalars"] = torch.nn.GaussianNLLLoss(eps=0.01)

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

        output = self.forward(single_source_batch)

        loss = self.compute_loss(output, single_source_batch)

        return loss, {"loss": loss}, single_source_batch["species"].shape[0]

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

        output = self.forward(single_source_batch)

        loss = self.compute_loss(output, single_source_batch)

        return loss, {"loss": loss}, single_source_batch["species"].shape[0]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """fit method for doing a predict step"""

        batch_clone = {}
        for k, v in batch.items():
            batch_clone[k] = v.clone().to(self.device)
        output = self.forward(batch_clone)

        detached_output: Any
        detached_output = {}
        for k, v in output.items():
            detached_output[k] = v.detach()

        del output
        del batch
        del batch_clone
        gc.collect()

        return detached_output
