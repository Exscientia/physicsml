# Allegro

The Allegro model as proposed in [Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics](https://arxiv.org/abs/2204.05249).

## Supervised

The base ``allegro_model`` as proposed in the original paper. It has the ability to output graph scalars, node scalars,
and a node vector. The model has the following config

```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_layers: int = 2``

    The number of Allegro layers.
* ``max_ell: int = 2``

    The maximum SO(3) irreps dimension to use in the model (during interactions)
* ``parity: bool = True``

    Whether to use parity odd features in the model
* ``mlp_irreps: str = "16x0e"``

    The output MLP irreps.
* ``mlp_latent_dimensions: List[int] = [128]``

    The MLP dimensions.
* ``latent_mlp_latent_dimensions: List[int] = [1024, 1024, 1024]``

    The latent MLP dimensions.
* ``env_embed_multiplicity: int = 32``

    The environment embedding multiplicity.
* ``two_body_latent_mlp_latent_dimensions: List[int] = [128, 256, 512, 1024]``

    The two body interaction MLP dimensions.
* ``num_bessel: int = 8``

    The highest order of bessel functions to use.
* ``bessel_basis_trainable: bool = True``

    Whether the bessel basis has trainable weights
* ``num_polynomial_cutoff: int = 6``

    The cut-off polynomial envelope power.
* ``avg_num_neighbours: Optional[float] = None``

    The average number of neighbours in the dataset.
* ``embed_initial_edge: bool = True``

    Whether to embed the initial edge features.
* ``latent_resnet: bool = True``

    Whether to use a latent resnet.
* ``scaling_mean: float = 0.0``

    The scaling mean of the output predictions
* ``scaling_std: float = 1.0``

    The scaling std of the output predictions.

* ``compute_forces: bool = False``

    Whether to compute forces as the gradient of the ``y_graph_scalars`` and use those as the ``y_node_vector`` output.
* ``y_node_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_scalars``.
* ``y_node_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_vector``.
* ``y_graph_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_graph_scalars``.
```

## Supervised mean and variance

The ``mean_var_allegro_model`` which predicts a mean and variance for the ``y_graph_scalars`` during training and inference.
It has the same comfig as the ``allegro_model`` but without the ability to specify a ``y_graph_scalars_loss_config`` which is
hardcoded to be the ``torch.nn.GaussianNLLLoss``. The model config is as follows

```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_layers: int = 2``

    The number of Allegro layers.
* ``max_ell: int = 2``

    The maximum SO(3) irreps dimension to use in the model (during interactions)
* ``parity: bool = True``

    Whether to use parity odd features in the model
* ``mlp_irreps: str = "16x0e"``

    The output MLP irreps.
* ``mlp_latent_dimensions: List[int] = [128]``

    The MLP dimensions.
* ``latent_mlp_latent_dimensions: List[int] = [1024, 1024, 1024]``

    The latent MLP dimensions.
* ``env_embed_multiplicity: int = 32``

    The environment embedding multiplicity.
* ``two_body_latent_mlp_latent_dimensions: List[int] = [128, 256, 512, 1024]``

    The two body interaction MLP dimensions.
* ``num_bessel: int = 8``

    The highest order of bessel functions to use.
* ``bessel_basis_trainable: bool = True``

    Whether the bessel basis has trainable weights
* ``num_polynomial_cutoff: int = 6``

    The cut-off polynomial envelope power.
* ``avg_num_neighbours: Optional[float] = None``

    The average number of neighbours in the dataset.
* ``embed_initial_edge: bool = True``

    Whether to embed the initial edge features.
* ``latent_resnet: bool = True``

    Whether to use a latent resnet.
* ``scaling_mean: float = 0.0``

    The scaling mean of the output predictions
* ``scaling_std: float = 1.0``

    The scaling std of the output predictions.

* ``compute_forces: bool = False``

    Whether to compute forces as the gradient of the ``y_graph_scalars`` and use those as the ``y_node_vector`` output.
* ``y_node_scalars_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_scalars``.
* ``y_node_vector_loss_config: Optional[Dict] = None``

    The loss config for the ``y_node_vector``.
```
