# Torch layer

The lowest layer is the ``torch`` layer. This consists of the ``torch`` dataset, dataloader, optimizers, and schedulers.

## Datasets

In the ``physicsml`` package most models are graph based, meaning that they can all share the same low-level graph dataset.
Once the featurised data is given to the ``train`` or ``predict`` methods, it is internally converted to a ``torch`` compatible
dataset. This dataset is built on top of ``torch-geometric`` (see their [docs](https://pytorch-geometric.readthedocs.io/en/latest/)),
an excellent library for graph-based torch modelling. In general, a model developer will rarely have to worry about this
(unless they are implementing a special kind of model that requires extra inputs).

The only exception to this is the ANI models. They require a special dataset (which is simpler) since the only inputs
they use are ``species`` and ``coordinates``.

## Dataloaders

Again, we use the ``torch-geometric`` dataloading for all models except for the ANI models. Model developers will
rarely have to modify this unless their models require additional inputs. Here, we will describe what a batch from each
dataloader looks like and what you need to expect as an input to your model.

### ``torch-geometric`` batch

A ``torch-geometric`` batch contains the following key-value pairs


```{toggle}
* ``num_nodes: Shape = torch.Size([]), Type = torch.int64``

    The total number of nodes in the batch.
* ``edge_index: Shape = torch.Size([2, num_edges]) Type = torch.int64.``

    The edge indices (receiver, sender). They are concatenated with a cumulative shift of the number of nodes in previous
    graphs (for batch GNN operations like scatter).
* ``node_attrs: Shape = torch.Size([num_nodes, dim_node_attrs]) Type = torch.float32``

    The node attributes (i.e. initial node features). It is a good idea to keep these distinct from node features
    (for downstream operations which require the original node attributes).
* ``edge_attrs: Shape = torch.Size([num_edges, dim_edge_attrs]) Type = torch.float32``

    The edge attributes (i.e. initial edge features). It is a good idea to keep these distinct from edge features
    (for downstream operations which require the original edge attributes).
* ``coordinates: Shape = torch.Size([num_nodes, 3]) Type = torch.float32``

    The coordinates.
* ``total_atomic_energy: Shape = torch.Size([num_graphs]) Type = torch.float32``

    The summed up atomic energies.
* ``raw_atomic_numbers: Shape = torch.Size([num_nodes]) Type = torch.int64``

    The raw mapped atomic numbers.
* ``atomic_numbers: Shape = torch.Size([num_nodes, num_elements]) Type = torch.float32``

    The one-hot encoded atomic numbers.
* ``cell: Shape = torch.Size([3, 3]) Type = torch.float32``

    The unit cell dimensions. This is used to compute distance and vectors when working in periodic systems.
* ``cell_shift_vector: Shape = torch.Size([num_edges, 3]) Type = torch.float32``

    The shift vectors for the connected edges used to compute distances and vectors when working in periodic systems.
* ``y_graph_scalars: Shape = torch.Size([num_graphs, y_graph_scalars_num_tasks]) Type = torch.float32``

    The graph level scalars references for the predictions.
* ``y_edge_scalars: Shape = torch.Size([num_edges, y_edge_scalars_num_tasks]) Type = torch.float32``

    The edge level scalars references for the predictions.
* ``y_node_scalars: Shape = torch.Size([num_nodes, y_node_scalars_num_tasks]) Type = torch.float32``

    The node level scalars references for the predictions.
* ``y_graph_vector: Shape = torch.Size([num_graphs, 3]) Type = torch.float32``

    The graph level vector references for the predictions.
* ``y_edge_vector: Shape = torch.Size([num_edge, 3]) Type = torch.float32``

    The edge level vector references for the predictions.
* ``y_node_vector: Shape = torch.Size([num_nodes, 3]) Type = torch.float32``

    The node level vector references for the predictions.
* ``num_graphs: Shape = torch.Size([]) Type = torch.int64``

    The number of graphs in the batch.
* ``batch: Shape = torch.Size([num_nodes]) Type = torch.int64``

    The batch index of each node (which graph it belongs to).
* ``ptr: Shape = torch.Size([num_graphs + 1]) Type = torch.int64``

    The cumulative number of nodes in each graph (with an extra 0 at the beginning).
```

An ANI model batch looks like

```{toggle}
* ``species: Shape = torch.Size([num_graphs, max_num_nodes]) Type = torch.int64``

    The mapped atom types padded up to the max number of nodes in the batch.
* ``coordinates: Shape = torch.Size([num_graphs, max_num_nodes, 3]) Type = torch.float32``

    The coordinates.
* ``total_atomic_energy: Shape = torch.Size([num_graphs]) Type = torch.float32``

    The summed up atomic energies.
* ``y_graph_scalars: Shape = torch.Size([num_graphs, y_graph_scalars_num_tasks]) Type = torch.float32``

    The graph level scalars references for the predictions.
* ``y_node_vector: Shape = torch.Size([num_nodes, 3]) Type = torch.float32``

    The node level vector references for the predictions.
```

By default the precision for floating point number is ``float32`` unless otherwise specified in the ``Trainer`` [config](lightning_layer.md#trainer-config)

## ``optimizer``

All optimizers from ``torch.optim`` can be used. We have created a serialisation for loading them via configs. The config
has the following structure

```{toggle}
* ``name``: The name of optimizer (from the ``torch.optim`` module).
* ``config``: A dict of the kwargs the optimizer takes.
```

## ``scheduler`` config

All schedulers from ``torch.optim.lr_scheduler`` can be used. We have created a serialisation for loading them via configs.
The config has the following structure

```{toggle}
* ``name``: The name of scheduler (from the ``torch.optim.lr_scheduler`` module).
* ``config``: A dict of the kwargs the scheduler takes.
* ``interval``: The interval to use (for schedulers like ``ReduceLROnPlateua``).
* ``frequency``: The interval frequency to use (for schedulers like ``ReduceLROnPlateua``).
* ``monitor``: The metric to monitor (for schedulers like ``ReduceLROnPlateua``).
* ``strict``: Whether to fail or raise a warning if monitoring metric not found (for schedulers like ``ReduceLROnPlateua``).
```
