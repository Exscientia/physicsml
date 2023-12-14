# MolFlux layer

The ``molflux`` layer handles all the interaction with ``physicsml`` models. All functionality for loading, saving, training, and
inference are handled through its standard API (this does not include interfaces for plugins such as OpenMM, see [plugins](../plugins/openmm.md)
for that).

For a complete overview of the ``molflux`` API, see its [documentation](https://exscientia.github.io/molflux/pages/standard_api/intro.html).
Here, we will discuss the relevant parts for the ``physicsml`` package.

The overall pipeline of any model build can be divided into 5 stages

## Accessing datasets

Datasets can be accessed via the ``molflux.datasets`` module. There are built in datasets that come available with ``molflux``
by default (see what is available [here](../datasets/qm_datasets.md)). You can also load preprocessed datasets from disk and remote storage
(such as ``s3``). If you would like to share your datasets with a wider audience, you can register them in the ``molflux.datasets``
module and make them available (see the ``molflux`` [documentation](https://exscientia.github.io/molflux/pages/datasets/intro.html)
on how to do that).

## Featurising datasets

To stay in line with the modular principles of ``molflux`` and ``physicsml``, we separate the featurisation of datasets
(e.g. extracting the atomic numbers and coordinates) into its own step. This not only allows for reducing redundant computation
but also makes the process of adding and expanding the available features more streamlined.

For more information about featurisation, see [PhysicsML features](../features/intro.md).

## Splitting datasets

The next stage is splitting the datasets for model evaluation and benchmarking. This is done via the ``molflux.splits`` module
(see docs [here](https://exscientia.github.io/molflux/pages/splits/intro.html)). If you already have pre-specified splits
for your datasets, then you do not have to worry about this.

## Model building

The main part of the pipeline is model building. We divide this section into model loading, model training, and model saving.

### Model loading

Models are best loaded by specifying a model config and using the ``molflux.modelzoo.load_from_dict`` function. The model
config has this generic form

```python
from molflux.modelzoo import load_from_dict

model_config = {
    "name": <name of model>,
    "config": {
        "x_features": <list of x features generated from featurisation>,
        "y_features": <list of y features to train on>,
        ...,  # a bunch of model specific kwargs,
        "optimizer": ...,  # the optimizer config
        "scheduler": ...,  # the scheduler config
        "datamodule": ...,  # a bunch of lightning specific kwargs to control the datamodule
        "trainer": ...,  # a bunch of lightning specific kwargs to control the training
        "transfer_learning": ...,  # a bunch of kwargs to control transfer learning
    }
}

model = load_from_dict(model_config)
```

This loads the model object (which is ``molflux`` a model that can handle the training and inference routines).

### Model training

Once the model is loaded, training is as simple as

```python
model.train(
    train_data=training_dataset,
    validation_data=validation_dataset,
)
```

The ``validation_data`` is optional (for early stopping and monitoring) and models can in principle be trained on a ``train_data``
only.

Under the hood, this sets up a bunch of objects from ``lightning`` to run the training. First, it instantiates the ``torch``
module (i.e. the actual model code), the datamodule, and the ``Trainer``. Then, it passes the module and the datamodule
to the ``Trainer`` to run the training. For more info, see [Lightning layer](lightning_layer.md).

```{note}
If a transfer learning config is specified, the training routine is a bit more involved. See [transfer learning](transfer_learning.md) for more info.
```

### Model saving

Once the model is trained, we can simply save it by doing

```python
from molflux.core import save_model

save_model(model, "path_to_model", featurisation_metadata)
```

It is important to persist the ``featurisation_metadata`` so that the model can be inferenced later on. Saving the model
creates a directory with the model config, featursation config, model artefacts (weights checkpoint), and a frozen requirements
file to recreate the environment it was trained in. For more information, see the ``molflux`` [docs](https://exscientia.github.io/molflux/pages/production/models.html).

## Computing metrics

Once the model is trained, we can compute some metrics. This is simply done by computing some predictions and using
the supplied metrics functionality of ``molflux``. See [here](https://exscientia.github.io/molflux/pages/metrics/intro.html)
for more info.
