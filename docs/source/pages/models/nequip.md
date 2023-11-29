# NequIP

The NequIP model as proposed in [E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials](https://arxiv.org/abs/2101.03164).

## Supervised

The base ``nequip_model`` as proposed in the original paper. It has the ability to output graph scalars, a graph vector,
node scalars, and a node vector. The model has the following config

```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_layers: int = 4``

    The number of layers of the model.
* ``max_ell: int = 2``

    The maximum SO(3) irreps dimension to use in the model (during interactions).
* ``parity: bool = True``

    Whether to use parity odd irreps.
* ``num_features: int = 32``

    The number of features of the hidden embeddings.
* ``mlp_irreps: str = "16x0e"``

    The output MLP irreps.
* ``num_bessel: int = 8``

    The number of bessel functions to use.
* ``bessel_basis_trainable: bool = True``

    Whether the bessel function weights are trainable.
* ``num_polynomial_cutoff: int = 6``

    The cut-off polynomial envelope power.
* ``self_connection: bool = True``

    Whether to use self connections
* ``resnet: bool = True``

    Whether to use a resnet.
* ``avg_num_neighbours: Optional[float] = None``

    The average number of neighbours in the dataset.
* ``scaling_mean: float = 0.0``

    The scaling mean of the model output.
* ``scaling_std: float = 1.0``

    The scaling std of the model output.
* ``compute_forces: bool = False``

    Whether to compute forces as the gradient of the ``y_graph_scalars`` and use those as the ``y_node_vector`` output.
* ``y_node_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_scalars``.
* ``y_node_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_edge_scalars``.
* ``y_graph_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_scalars``.
* ``y_graph_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_vector``.
```

## Supervised mean and variance

The ``mean_var_nequip_model`` which predicts a mean and variance for the ``y_graph_scalars`` during training and inference.
It has the same comfig as the ``nequip_model`` but without the ability to specify a ``y_graph_scalars_loss_config`` which is
hardcoded to be the ``torch.nn.GaussianNLLLoss``. The model config is as follows

```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_layers: int = 4``

    The number of layers of the model.
* ``max_ell: int = 2``

    The maximum SO(3) irreps dimension to use in the model (during interactions).
* ``parity: bool = True``

    Whether to use parity odd irreps.
* ``num_features: int = 32``

    The number of features of the hidden embeddings.
* ``mlp_irreps: str = "16x0e"``

    The output MLP irreps.
* ``num_bessel: int = 8``

    The number of bessel functions to use.
* ``bessel_basis_trainable: bool = True``

    Whether the bessel function weights are trainable.
* ``num_polynomial_cutoff: int = 6``

    The cut-off polynomial envelope power.
* ``self_connection: bool = True``

    Whether to use self connections
* ``resnet: bool = True``

    Whether to use a resnet.
* ``avg_num_neighbours: Optional[float] = None``

    The average number of neighbours in the dataset.
* ``scaling_mean: float = 0.0``

    The scaling mean of the model output.
* ``scaling_std: float = 1.0``

    The scaling std of the model output.
* ``compute_forces: bool = False``

    Whether to compute forces as the gradient of the ``y_graph_scalars`` and use those as the ``y_node_vector`` output.
* ``y_node_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_scalars``.
* ``y_node_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_edge_scalars``.
* ``y_graph_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_vector``.
```

## Supervised feature scale-shift

The ``ssf_nequip_model`` which adds scale and shift for the hidden embeddings in the MACE model as proposed in
[Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning](https://arxiv.org/abs/2210.08823). This is
useful for transfer learning. The config is the same as the ``nequip_model``. The model config is as follows

```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_layers: int = 4``

    The number of layers of the model.
* ``max_ell: int = 2``

    The maximum SO(3) irreps dimension to use in the model (during interactions).
* ``parity: bool = True``

    Whether to use parity odd irreps.
* ``num_features: int = 32``

    The number of features of the hidden embeddings.
* ``mlp_irreps: str = "16x0e"``

    The output MLP irreps.
* ``num_bessel: int = 8``

    The number of bessel functions to use.
* ``bessel_basis_trainable: bool = True``

    Whether the bessel function weights are trainable.
* ``num_polynomial_cutoff: int = 6``

    The cut-off polynomial envelope power.
* ``self_connection: bool = True``

    Whether to use self connections
* ``resnet: bool = True``

    Whether to use a resnet.
* ``avg_num_neighbours: Optional[float] = None``

    The average number of neighbours in the dataset.
* ``scaling_mean: float = 0.0``

    The scaling mean of the model output.
* ``scaling_std: float = 1.0``

    The scaling std of the model output.
* ``compute_forces: bool = False``

    Whether to compute forces as the gradient of the ``y_graph_scalars`` and use those as the ``y_node_vector`` output.
* ``y_node_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_scalars``.
* ``y_node_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_edge_scalars``.
* ``y_graph_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_scalars``.
* ``y_graph_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_vector``.
```

## Supervised adapters

The ``adapter_nequip_model`` which adds adapters at the beginning and end of the message passing backbone of the NequIP model
as proposed in [AdapterGNN](https://arxiv.org/abs/2304.09595). This is useful for transfer learning. The config is the
same as the ``nequip_model`` but with the added kwargs for ``ratio_adapter_down`` and ``initial_s``. The model config is
as follows

```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_layers: int = 4``

    The number of layers of the model.
* ``max_ell: int = 2``

    The maximum SO(3) irreps dimension to use in the model (during interactions).
* ``parity: bool = True``

    Whether to use parity odd irreps.
* ``num_features: int = 32``

    The number of features of the hidden embeddings.
* ``mlp_irreps: str = "16x0e"``

    The output MLP irreps.
* ``num_bessel: int = 8``

    The number of bessel functions to use.
* ``bessel_basis_trainable: bool = True``

    Whether the bessel function weights are trainable.
* ``num_polynomial_cutoff: int = 6``

    The cut-off polynomial envelope power.
* ``self_connection: bool = True``

    Whether to use self connections
* ``resnet: bool = True``

    Whether to use a resnet.
* ``avg_num_neighbours: Optional[float] = None``

    The average number of neighbours in the dataset.
* ``scaling_mean: float = 0.0``

    The scaling mean of the model output.
* ``scaling_std: float = 1.0``

    The scaling std of the model output.
* ``compute_forces: bool = False``

    Whether to compute forces as the gradient of the ``y_graph_scalars`` and use those as the ``y_node_vector`` output.
* ``y_node_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_scalars``.
* ``y_node_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_edge_scalars``.
* ``y_graph_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_scalars``.
* ``y_graph_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_vector``.
```
