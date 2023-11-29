# EGNN

The EGNN model as proposed in [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844).

## Supervised

The base ``egnn_model`` as proposed in the original paper. It has the ability to output graph scalars, edge scalars,
node scalars, and optionally a node vector as the gradient of the graph scalars (e.g. forces). The model has the
following config

```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_rbf: int = 0``

    The number of radial bessel functions to use (0 in the original paper).
* ``num_layers: int = 4``

    The number of message passing layers.
* ``num_layers_phi: int = 2``

    The number of layers in the message passing MLPs.
* ``num_layers_pooling: int = 2``

    The number of layers in the pooling MLPs.
* ``c_hidden: int = 128``

    The hidden dimension of the model.
* ``modify_coords: bool = False``

    Whether to modify the coordinates during message passing.
* ``jitter: Optional[float] = None``

    Whether to randomly jitter the atom coordinates before the forward pass.
* ``pool_type: Literal["sum", "mean"] = "sum"``

    Whether to pool by summing or taking the mean.
* ``pool_from: Literal["nodes", "nodes_edges", "edges"] = "nodes"``

    Whether to pool from the node embeddings, edge embeddings, or both.
* ``dropout: Optional[float] = None``

    Whether to use dropout during training.
* ``mlp_activation: Optional[str] = "SiLU"``

    The activation functions of the MLPs.
* ``mlp_output_activation: Optional[str] = None``

    The activation function of the MLP outputs.
* ``output_activation: Optional[str] = None``

    The activation function of the model output.
* ``scaling_mean: float = 0.0``

    The scaling mean of the model output.
* ``scaling_std: float = 1.0``

    The scaling std of the model output.
* ``compute_forces: bool = False``

    Whether to compute forces as the gradient of the ``y_graph_scalars`` and use those as the ``y_node_vector`` output.
* ``y_node_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_scalars``.
* ``y_node_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_vector``.
* ``y_edge_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_edge_scalars``.
* ``y_graph_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_scalars``.
```

## Supervised mean and variance

The ``mean_var_egnn_model`` which predicts a mean and variance for the ``y_graph_scalars`` during training and inference.
It has the same comfig as the ``egnn_model`` but without the ability to specify a ``y_graph_scalars_loss_config`` which is
hardcoded to be the ``torch.nn.GaussianNLLLoss``. The model config is as follows

```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_rbf: int = 0``

    The number of radial bessel functions to use (0 in the original paper).
* ``num_layers: int = 4``

    The number of message passing layers.
* ``num_layers_phi: int = 2``

    The number of layers in the message passing MLPs.
* ``num_layers_pooling: int = 2``

    The number of layers in the pooling MLPs.
* ``c_hidden: int = 128``

    The hidden dimension of the model.
* ``modify_coords: bool = False``

    Whether to modify the coordinates during message passing.
* ``jitter: Optional[float] = None``

    Whether to randomly jitter the atom coordinates before the forward pass.
* ``pool_type: Literal["sum", "mean"] = "sum"``

    Whether to pool by summing or taking the mean.
* ``pool_from: Literal["nodes", "nodes_edges", "edges"] = "nodes"``

    Whether to pool from the node embeddings, edge embeddings, or both.
* ``dropout: Optional[float] = None``

    Whether to use dropout during training.
* ``mlp_activation: Optional[str] = "SiLU"``

    The activation functions of the MLPs.
* ``mlp_output_activation: Optional[str] = None``

    The activation function of the MLP outputs.
* ``output_activation: Optional[str] = None``

    The activation function of the model output.
* ``scaling_mean: float = 0.0``

    The scaling mean of the model output.
* ``scaling_std: float = 1.0``

    The scaling std of the model output.
* ``compute_forces: bool = False``

    Whether to compute forces as the gradient of the ``y_graph_scalars`` and use those as the ``y_node_vector`` output.
* ``y_node_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_scalars``.
* ``y_node_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_vector``.
* ``y_edge_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_edge_scalars``.
```

## Supervised feature scale-shift

The ``ssf_egnn_model`` which adds scale and shift for the hidden embeddings in the EGNN model as proposed in
[Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning](https://arxiv.org/abs/2210.08823). This is
useful for transfer learning. The config is the same as the ``egnn_model``. The model config is as follows

```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_rbf: int = 0``

    The number of radial bessel functions to use (0 in the original paper).
* ``num_layers: int = 4``

    The number of message passing layers.
* ``num_layers_phi: int = 2``

    The number of layers in the message passing MLPs.
* ``num_layers_pooling: int = 2``

    The number of layers in the pooling MLPs.
* ``c_hidden: int = 128``

    The hidden dimension of the model.
* ``modify_coords: bool = False``

    Whether to modify the coordinates during message passing.
* ``jitter: Optional[float] = None``

    Whether to randomly jitter the atom coordinates before the forward pass.
* ``pool_type: Literal["sum", "mean"] = "sum"``

    Whether to pool by summing or taking the mean.
* ``pool_from: Literal["nodes", "nodes_edges", "edges"] = "nodes"``

    Whether to pool from the node embeddings, edge embeddings, or both.
* ``dropout: Optional[float] = None``

    Whether to use dropout during training.
* ``mlp_activation: Optional[str] = "SiLU"``

    The activation functions of the MLPs.
* ``mlp_output_activation: Optional[str] = None``

    The activation function of the MLP outputs.
* ``output_activation: Optional[str] = None``

    The activation function of the model output.
* ``scaling_mean: float = 0.0``

    The scaling mean of the model output.
* ``scaling_std: float = 1.0``

    The scaling std of the model output.
* ``compute_forces: bool = False``

    Whether to compute forces as the gradient of the ``y_graph_scalars`` and use those as the ``y_node_vector`` output.
* ``y_node_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_scalars``.
* ``y_node_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_vector``.
* ``y_edge_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_edge_scalars``.
* ``y_graph_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_scalars``.
```

## Supervised adapters

The ``adapter_egnn_model`` which adds adapters at the beginning and end of the message passing backbone of the EGNN model
as proposed in [AdapterGNN](https://arxiv.org/abs/2304.09595). This is useful for transfer learning. The config is the
same as the ``egnn_model`` but with the added kwargs for ``ratio_adapter_down`` and ``initial_s``. The model config is
as follows

```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_rbf: int = 0``

    The number of radial bessel functions to use (0 in the original paper).
* ``num_layers: int = 4``

    The number of message passing layers.
* ``num_layers_phi: int = 2``

    The number of layers in the message passing MLPs.
* ``num_layers_pooling: int = 2``

    The number of layers in the pooling MLPs.
* ``c_hidden: int = 128``

    The hidden dimension of the model.
* ``ratio_adapter_down: int = 10``

    The ratio of the down layer in the adapters.
* ``initial_s: float = 0.01``

    The starting value of the s parameter for combining the outputs from the adapters and the message passing backbone.
* ``modify_coords: bool = False``

    Whether to modify the coordinates during message passing.
* ``jitter: Optional[float] = None``

    Whether to randomly jitter the atom coordinates before the forward pass.
* ``pool_type: Literal["sum", "mean"] = "sum"``

    Whether to pool by summing or taking the mean.
* ``pool_from: Literal["nodes", "nodes_edges", "edges"] = "nodes"``

    Whether to pool from the node embeddings, edge embeddings, or both.
* ``dropout: Optional[float] = None``

    Whether to use dropout during training.
* ``mlp_activation: Optional[str] = "SiLU"``

    The activation functions of the MLPs.
* ``mlp_output_activation: Optional[str] = None``

    The activation function of the MLP outputs.
* ``output_activation: Optional[str] = None``

    The activation function of the model output.
* ``scaling_mean: float = 0.0``

    The scaling mean of the model output.
* ``scaling_std: float = 1.0``

    The scaling std of the model output.
* ``compute_forces: bool = False``

    Whether to compute forces as the gradient of the ``y_graph_scalars`` and use those as the ``y_node_vector`` output.
* ``y_node_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_scalars``.
* ``y_node_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_vector``.
* ``y_edge_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_edge_scalars``.
* ``y_graph_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_scalars``.
```
