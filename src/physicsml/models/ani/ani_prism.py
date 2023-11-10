from typing import TYPE_CHECKING, Any, Dict, List, Union

import datasets
from molflux.features.info import RepresentationInfo
from molflux.modelzoo.models.lightning.config import DataModuleConfig, TrainerConfig

from lightning.pytorch import LightningModule
from physicsml.models.prism import PhysicsMLPrismBase

if TYPE_CHECKING:
    import torch

_DESCRIPTION = """
A representation is generated from the latent layers of the ANI model
"""


class ANIFeaturiser(LightningModule):
    def __init__(self, module: Any, which_rep: str) -> None:
        super().__init__()

        self.module = module.aev_computer
        self.which_rep = which_rep

    def forward(self, x: Dict[str, Any]) -> Any:
        out = self.module((x["species"], x["coordinates"]))

        if self.which_rep == "aev":
            list_of_features = out.aevs.tolist()

        elif self.which_rep == "aev_sum":
            list_of_features = out.aevs.sum(1).tolist()

        else:
            raise RuntimeError(f"which_rep = {self.which_rep} is unknown")

        return list_of_features

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """fit method for doing a validation step"""

        for k, v in batch.items():
            batch[k] = v.to(self.device)

        output = self.forward(batch)

        return output


class ANI(PhysicsMLPrismBase):
    def __init__(
        self,
        which_rep: str = "aev_sum",
        **kwargs: Any,
    ) -> None:
        """

        Args:
            path:
            which_rep: which representation to extract from model. Choose from 'aev', 'aev_sum'. default = "aev_sum".
            **kwargs:
        """
        super().__init__(**kwargs)

        _allowed_reps = [
            "aev",
            "aev_sum",
        ]

        self.model.module = ANIFeaturiser(self.model.module, which_rep=which_rep)

    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _extract_features(
        self,
        dataset_feated: datasets.Dataset,
        datamodule_config: Union[DataModuleConfig, Dict[str, Any], None] = None,
        trainer_config: Union[TrainerConfig, Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> List:
        """

        Takes 2D molecules (as SMILES, OEMols or OEMolBytes).

        Args:
            samples: The sample molecules to be featurised
            **kwargs:

        Returns: a dict of the features as torch tensors
        """

        batched_features: List[torch.Tensor] = self.model._predict_batched(
            data=dataset_feated,
            datamodule_config=datamodule_config,
            trainer_config=trainer_config,
        )

        features = [xs for x in batched_features for xs in x]

        return features
