import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import datasets
import molflux.core as molflux_core
import torch
from molflux.features.bases import RepresentationBase
from molflux.features.typing import MolArray
from molflux.modelzoo.models.lightning.config import DataModuleConfig, TrainerConfig

from physicsml.backends.backend_selector import to_mol, to_mol_bytes
from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.utils import load_from_dvc

if TYPE_CHECKING:
    from physicsml.lightning.model import PhysicsMLModelBase

datasets.disable_progress_bar()
logger = logging.getLogger(__name__)


class PhysicsMLPrismBase(RepresentationBase):
    def __init__(
        self,
        path: Optional[str] = None,
        rev: Optional[str] = None,
        repo_url: Optional[str] = None,
        model_path_in_repo: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        del kwargs

        if not (
            (path is not None)
            ^ (
                (rev is not None)
                and (repo_url is not None)
                and (model_path_in_repo is not None)
            )
        ):
            raise ValueError(
                "Must specify either 'path' or 'rev', 'repo_url', 'model_path_in_repo' (but not both).",
            )

        if path is not None:
            self.model: PhysicsMLModelBase = molflux_core.load_model(path)  # type: ignore
            self.featurisation_metadata = molflux_core.load_featurisation_metadata(
                f"{path.rstrip('/')}/featurisation_metadata.json",
            )
        elif (
            (repo_url is not None)
            and (rev is not None)
            and (model_path_in_repo is not None)
        ):
            self.model, self.featurisation_metadata = load_from_dvc(
                repo_url=repo_url,
                rev=rev,
                model_path_in_repo=model_path_in_repo,
            )
        else:
            raise RuntimeError("Could not load model")

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

    @abstractmethod
    def _extract_features(
        self,
        dataset_feated: datasets.Dataset,
        datamodule_config: Union[DataModuleConfig, Dict[str, Any], None] = None,
        trainer_config: Union[TrainerConfig, Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> List:
        """method to extract features from model."""

    def _featurise(
        self,
        samples: MolArray,
        datamodule_config: Union[DataModuleConfig, Dict[str, Any], None] = None,
        trainer_config: Union[TrainerConfig, Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """

        Takes 2D molecules (as SMILES, OEMols or OEMolBytes).

        Args:
            samples: The sample molecules to be featurised

        Returns: a dict of the features as torch tensors
        """

        samples = [to_mol(self.backend)(sample) for sample in samples]

        total_len = len(samples)
        failed_idxs = []
        for idx, mol in enumerate(samples):
            if mol is None:
                failed_idxs.append(idx)
                logger.warning(
                    f"Molecule at index {idx} failed to convert to bytes and will return None features.",
                )

        samples = [to_mol_bytes(self.backend)(x) for x in samples if x is not None]

        dataset = datasets.Dataset.from_dict({"tmp_mol": samples})
        dataset_feated = molflux_core.featurise_dataset(
            dataset,
            self.featurisation_metadata,
            num_proc=4,
            batch_size=128,
        )

        features = self._extract_features(
            dataset_feated=dataset_feated,
            datamodule_config=datamodule_config,
            trainer_config=trainer_config,
        )

        for idx in sorted(failed_idxs):
            features.insert(idx, None)

        assert len(features) == total_len

        return {self.tag: features}


class PrismLightningBase(PhysicsMLModuleBase):
    def __init__(self) -> None:
        super().__init__(model_config=None)  # type: ignore

    def compute_loss(self, input: Any, target: Any) -> Dict[str, torch.Tensor]:
        return {"loss": torch.empty(0)}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """method for doing a predict step"""

        batch_dict = self.graph_batch_to_batch_dict(batch)
        output = self(batch_dict)

        return output
