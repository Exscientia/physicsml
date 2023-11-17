from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import datasets
from molflux.features.info import RepresentationInfo
from molflux.modelzoo.models.lightning.config import DataModuleConfig, TrainerConfig
from torch_geometric.utils.scatter import scatter

from physicsml.models.prism import PhysicsMLPrismBase, PrismLightningBase

if TYPE_CHECKING:
    import torch

_DESCRIPTION = """
A representation is generated from the latent layers of the MACE model
"""


class MACEFeaturiser(PrismLightningBase):
    def __init__(self, module: Any, which_rep: str, which_block: int) -> None:
        super().__init__()

        self.module = module
        self.which_rep = which_rep
        self.which_block = which_block

    def forward(self, x: Dict[str, Any]) -> Any:
        x = self.module(x)

        node_feats = x[f"node_feats_{self.which_block}"]
        node_feats = node_feats[:, : self.module.hidden_irreps[0].dim]
        batch = x["batch"]
        num_graphs = int(x["num_graphs"])

        list_of_features = []
        if self.which_rep == "node_embedding":
            for batch_id in batch.unique():
                batch_id_mask = batch == batch_id
                batch_id_node_feats = node_feats[batch_id_mask].tolist()
                list_of_features.append(batch_id_node_feats)

        elif self.which_rep == "graph_embedding_sum":
            pooled_node_feats = scatter(
                src=node_feats,
                index=batch,
                dim=0,
                dim_size=num_graphs,
                reduce="sum",
            )
            list_of_features = pooled_node_feats.tolist()

        elif self.which_rep == "graph_embedding_mean":
            pooled_node_feats = scatter(
                src=node_feats,
                index=batch,
                dim=0,
                dim_size=num_graphs,
                reduce="mean",
            )
            list_of_features = pooled_node_feats.tolist()

        else:
            raise RuntimeError(f"which_rep = {self.which_rep} is unknown")

        return list_of_features


class MACE(PhysicsMLPrismBase):
    def __init__(
        self,
        which_rep: str = "graph_embedding_mean",
        which_block: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            path:
            which_rep: which representation to extract from model. Choose from 'node_embedding', 'graph_embedding_sum', 'graph_embedding_mean', 'attention'. default = "graph_embedding_mean".
            which_block: where to slice the model (how many blocks to use). choose from 0 to num_blocks. default = None (all blocks).
            **kwargs:
        """
        super().__init__(**kwargs)

        _allowed_reps = [
            "node_embedding",
            "graph_embedding_sum",
            "graph_embedding_mean",
        ]

        assert which_rep in _allowed_reps, KeyError(
            f"{which_rep} option unknown, try one of {_allowed_reps}",
        )

        if which_block is None:
            which_block = len(self.model.module.mace.interactions) - 1

        self.model.module = MACEFeaturiser(
            module=self.model.module.mace,
            which_rep=which_rep,
            which_block=which_block,
        )

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
