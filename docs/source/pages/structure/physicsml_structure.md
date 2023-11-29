# PhysicsML structure

The ``physicsml`` package is built on top of the ``molflux`` ecosystem which provides a standard API for handling the
different stages of machine learning pipelines (accessing datasets, featurising, splitting, building models, and computing
metrics).

Apart from that, it also handles and abstracts away the boilerplate code for training deeplearning models
via the ``lighthing`` integration ``molflux[lightning]``. For more info about ``lightning`` see LINK.


In general, there are three layers to ``physicsml`` models:

1) The [``molflux``](molflux_layer.md) layer
2) The [``lightning``](lightning_layer.md)) layer
3) The [``torch``](torch_layer.md)) layer

In the following sections, we will discuss each of these layers, explain what they're responsible for, their flexibility,
and how you work with them.
