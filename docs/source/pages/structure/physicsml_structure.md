# PhysicsML structure

The ``physicsml`` package is built on top of the ``molflux`` [ecosystem](https://exscientia.github.io/molflux/index.html)
which provides a standard API for handling the different stages of machine learning pipelines (accessing datasets,
featurising, splitting, building models, and computing metrics).

``molflux`` also handles and abstracts away the boilerplate code for training ``torch`` models
via the ``lightning`` integration ``molflux[lightning]``. For more info about ``lightning``, see their [documentation](https://lightning.ai/pytorch-lightning).

## Layers

There are three layers to ``physicsml`` models:

1) The [``molflux``](molflux_layer.md) layer: handles all functionality relating to the high level API.
2) The [``lightning``](lightning_layer.md) layer: handles all boilerplate code for training and inference.
3) The [``torch``](torch_layer.md) layer: handles the ``torch`` level functionality of the models.

In the following sections, we discuss each of these layers, explain what they're responsible for, their flexibility,
and how you work with them.
