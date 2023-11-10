from typing import Dict, Literal, Optional

from pydantic.dataclasses import dataclass

from physicsml.lightning.config import ConfigDict, PhysicsMLModelConfig


@dataclass(config=ConfigDict)
class SSFEGNNModelConfig(PhysicsMLModelConfig):
    num_node_feats: int = 0
    num_edge_feats: int = 0
    num_rbf: int = 0
    num_layers: int = 4
    num_layers_phi: int = 2
    num_layers_pooling: int = 2
    c_hidden: int = 128
    modify_coords: bool = False
    pool_type: Literal["sum", "mean"] = "sum"
    pool_from: Literal["nodes", "nodes_edges", "edges"] = "nodes"
    dropout: Optional[float] = None
    mlp_activation: Optional[str] = "SiLU"
    mlp_output_activation: Optional[str] = None
    output_activation: Optional[str] = None
    scaling_mean: float = 0.0
    scaling_std: float = 1.0
    y_node_scalars_loss_config: Optional[Dict] = None
    y_edge_scalars_loss_config: Optional[Dict] = None
    y_graph_scalars_loss_config: Optional[Dict] = None
    y_node_vector_loss_config: Optional[Dict] = None
