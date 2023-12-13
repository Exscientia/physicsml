# PhysicsML models

The ``physicsml`` package has implementations of some of the most popular state-of-the-art 3d neural networks. These implementations
have been taken from their original repos. To fit into the standardised API, some of the code has been refactored.

```{important}
All the models have been checked to reproduce the same output at each layer as the code from the original repos. All the
models have been checked for similar training and inference speeds to the original repos.

As part of the refactor, some modules have been renamed and reorganised for better clarity.
```

Models can come in multiple flavours that share the same backbone architecture. The original models are called ``*_model``
whereas the extensions are prefixed with the appropriate prefix.
