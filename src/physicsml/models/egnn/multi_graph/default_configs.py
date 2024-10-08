from dataclasses import field
from typing import Literal

from pydantic.v1 import dataclasses

from physicsml.lightning.config import ConfigDict, PhysicsMLModelConfig


@dataclasses.dataclass(config=ConfigDict)
class MultiGraphEGNNModelConfig(PhysicsMLModelConfig):
    graph_names: list = field(
        default_factory=lambda: ["protein", "ligand", "ligand_pocket"],
    )
    num_node_feats: int = 0
    num_edge_feats: int = 0
    num_rbf: int = 0
    num_layers: int = 4
    num_layers_phi: int = 2
    num_layers_pooling: int = 2
    c_hidden: int = 128
    modify_coords: bool = False
    jitter: float | None = None
    pooling_head: Literal[
        "LigandPocketDiffPoolingHead",
        "LigandPocketPoolingHead",
        "InvariantLigandPocketPoolingHead",
    ] = "LigandPocketPoolingHead"
    pool_type: Literal["sum", "mean"] = "sum"
    pool_from: Literal["nodes", "nodes_edges", "edges"] = "nodes"
    dropout: float | None = None
    mlp_activation: str | None = "SiLU"
    mlp_output_activation: str | None = None
    output_activation: str | None = None
    scaling_mean: float = 0.0
    scaling_std: float = 1.0
    y_node_scalars_loss_config: dict | None = None
    y_edge_scalars_loss_config: dict | None = None
    y_graph_scalars_loss_config: dict | None = None
    y_node_vector_loss_config: dict | None = None
