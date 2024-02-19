import gc
import os
import shutil
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
from molflux.modelzoo.models.lightning.module import (
    LightningModuleBase,
    SingleBatchStepOutput,
)
from torch_geometric.data.batch import Batch

from physicsml.lightning.config import PhysicsMLModelConfig


class PhysicsMLModuleBase(
    LightningModuleBase,
):
    def __init__(self, model_config: PhysicsMLModelConfig) -> None:
        super().__init__(model_config=model_config)
        self.model_config = model_config

    @abstractmethod
    def compute_loss(self, input: Any, target: Any) -> Dict[str, torch.Tensor]:
        """a method to compute the loss for a model"""

    def compute_forces_by_gradient(
        self,
        energy: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        # compute forces as gradient of energy
        grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
        gradient = torch.autograd.grad(
            outputs=[energy],  # [n_graphs, ]
            inputs=[coordinates],  # [n_nodes, 3]
            grad_outputs=grad_outputs,  # type: ignore
            retain_graph=self.training,  # Make sure the graph is not destroyed during training
            create_graph=self.training,  # Create graph for second derivative
            allow_unused=True,
        )[
            0
        ]  # [n_nodes, 3]

        if gradient is None:
            raise RuntimeWarning("Gradient is None")
        forces = -1 * gradient

        return forces

    def graph_batch_to_batch_dict(self, graph_batch: Batch) -> Dict[str, torch.Tensor]:
        batch_dict: Dict[str, Any] = graph_batch.to_dict()
        batch_dict["num_graphs"] = torch.tensor(graph_batch.num_graphs)
        batch_dict["num_nodes"] = torch.tensor(batch_dict["num_nodes"])

        for k, v in batch_dict.items():
            if not isinstance(v, torch.Tensor):
                batch_dict[k] = torch.tensor(v)

            tensor = batch_dict[k]
            if torch.is_floating_point(tensor):
                batch_dict[k] = tensor.type(self.dtype)

        return batch_dict

    def _training_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: Optional[str],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        """fit method for doing a training step"""

        batch_dict = self.graph_batch_to_batch_dict(single_source_batch)

        if self.compute_forces:
            batch_dict["coordinates"].requires_grad = True
            output = self(batch_dict)
            forces = self.compute_forces_by_gradient(
                energy=output["y_graph_scalars"],
                coordinates=batch_dict["coordinates"],
            )
            if output.get("y_node_vector", None) is not None:
                raise RuntimeError(
                    "Cannot compute forces is model already outputs y_node_vector.",
                )

            output["y_node_vector"] = forces
        else:
            output = self(batch_dict)

        loss_dict = self.compute_loss(output, batch_dict)

        return loss_dict["loss"], loss_dict, single_source_batch.ptr.shape[0] - 1

    def _validation_step_on_single_source_batch(
        self,
        single_source_batch: Any,
        source_name: Optional[str],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> SingleBatchStepOutput:
        """fit method for doing a validation step"""

        batch_dict = self.graph_batch_to_batch_dict(single_source_batch)

        if self.compute_forces:
            with torch.enable_grad():
                batch_dict["coordinates"].requires_grad = True
                output = self(batch_dict)
                forces = self.compute_forces_by_gradient(
                    energy=output["y_graph_scalars"],
                    coordinates=batch_dict["coordinates"],
                )
                if output.get("y_node_vector", None) is not None:
                    raise RuntimeError(
                        "Cannot compute forces is model already outputs y_node_vector.",
                    )

                output["y_node_vector"] = forces
        else:
            output = self(batch_dict)

        loss_dict = self.compute_loss(output, batch_dict)

        return loss_dict["loss"], loss_dict, single_source_batch.ptr.shape[0] - 1

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """method for doing a predict step"""

        if self.compute_forces:
            with torch.inference_mode(False):
                batch_clone = batch.clone()
                batch_dict = self.graph_batch_to_batch_dict(batch_clone)
                batch_dict["coordinates"].requires_grad = True
                output = self(batch_dict)
                forces = self.compute_forces_by_gradient(
                    energy=output["y_graph_scalars"],
                    coordinates=batch_dict["coordinates"],
                )
                if output.get("y_node_vector", None) is not None:
                    raise RuntimeError(
                        "Cannot compute forces is model already outputs y_node_vector.",
                    )

                output["y_node_vector"] = forces
        else:
            batch_clone = batch.clone()
            batch_dict = self.graph_batch_to_batch_dict(batch_clone)
            output = self(batch_dict)

        detached_output: Any
        # if tensor
        if isinstance(output, torch.Tensor):
            detached_output = output.detach()
        # elif dict of tensors
        elif isinstance(output, dict) and all(
            isinstance(x, torch.Tensor) for x in output.values()
        ):
            detached_output = {}
            for k, v in output.items():
                detached_output[k] = v.detach()
        else:
            detached_output = output

        del output
        del batch
        del batch_dict
        del batch_clone
        gc.collect()

        return detached_output

    def on_fit_end(self) -> None:
        if self.trainer.global_rank == 0:
            for name in ["train", "validation"]:
                if os.path.exists(f"pre_batched_{name}_dataset"):
                    shutil.rmtree(f"pre_batched_{name}_dataset")

    def on_predict_end(self) -> None:
        if self.trainer.global_rank == 0:
            for name in ["train", "validation"]:
                if os.path.exists(f"pre_batched_{name}_dataset"):
                    shutil.rmtree(f"pre_batched_{name}_dataset")
