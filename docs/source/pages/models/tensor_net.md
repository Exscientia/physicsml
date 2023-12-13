# TensorNet

The TensorNet model as proposed in [TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials
](https://arxiv.org/abs/2306.06482).

## Supervised

The base ``tesnor_net_model`` as proposed in the original paper. It has the ability to output graph scalars,
node scalars, and optionally a node vector as the gradient of the graph scalars. The model has the following config


```{toggle}
* ``num_node_feats: int``

    The number of node features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_edge_feats: int``

    The number of edge features. Must be equal to the initial node feature dimension (sum of one-hot encoded features).
* ``num_features: int = 256``

    The number of features in the model.
* ``num_radial: int = 64``

    The number of radial bessel functions to use.
* ``num_interaction_layers: int = 3``

    The number of interaction layers.
* ``embedding_mlp_hidden_dims: List[int] = [512]``

    The embedding MLP's hidden dimensions.
* ``interaction_mlp_hidden_dims: List[int] = [256, 512]``

    The interaction MLP's hidden dimensions.
* ``scalar_output_mlp_hidden_dims: List[int] = [256, 128]``

    The scalar output MLP's hidden dimensions.
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
```
