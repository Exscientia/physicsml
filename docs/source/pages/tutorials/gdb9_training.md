---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# QM9 training

In this tutorial we provide a simple example of training the ANI model on the QM9 dataset. It will take you through
the steps required from access the dataset to featurisation and model training. The ``physicsml`` package is built as
an extension to ``molflux`` and so we will be mainly importing functionality from there. Check out the
``molflux`` [docs](https://exscientia.github.io/molflux/index.html) for more info!

We also require the ``rdkit`` package to handle the molecules and extract molecular information like atomic numbers and
coordinates, so make sure to ``pip install 'physicsml[rdkit]'`` to follow along!


## Loading the QM9 dataset

First, let's load a truncated QM9 dataset with 1000 datapoints. For more information on the
loading and using dataset, see the ``molflux`` [documentation](https://exscientia.github.io/molflux/pages/datasets/basic_usage.html).

```{code-cell} ipython3
from molflux.datasets import load_dataset_from_store

dataset = load_dataset_from_store("gdb9_trunc.parquet")

print(dataset)
```

````{note}
The dataset above is a truncated version to run efficiently in the docs. For running this locally, load the entire dataset
by doing

```{code-block} python
from molflux.datasets import load_dataset

dataset = load_dataset("gdb9", "rdkit")
```
````

The QM9 dataset contains multiple computed quantum mechanical properties of small molecules. For more information on
the individual properties, visit the [original paper](https://www.nature.com/articles/sdata201422). Here, we will focus
on the ``u0`` property which is the total atomic energy. You can also see that there is the ``mol_bytes`` column
which is the [``rdkit`` serialisation](https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol.ToBinary)
of the 3d molecules.


## Featurising

Next, we will featurise the dataset. The ANI model requires only the atomic numbers, coordinates, and atomic self energies.
For more information on the ``physicsml`` features, see [here](../features/intro.md).

```{code-cell} ipython3
import logging
logging.disable(logging.CRITICAL)

from molflux.core import featurise_dataset

featurisation_metadata = {
    "version": 1,
    "config": [
        {
            "column": "mol_bytes",
            "representations": [
                {
                    "name": "physicsml_features",
                    "config": {
                        "atomic_number_mapping": {
                            1: 0,
                            6: 1,
                            7: 2,
                            8: 3,
                            9: 4,
                        },
                        "atomic_energies": {
                            1: -0.6019805629746086,
                            6: -38.07749583990695,
                            7: -54.75225433326539,
                            8: -75.22521603087064,
                            9: -99.85134426752529
                        },
                        "backend": "rdkit",
                    },
                    "as": "{feature_name}"
                }
            ]
        }
    ]
}

# featurise the mols
featurised_dataset = featurise_dataset(
    dataset,
    featurisation_metadata=featurisation_metadata,
    num_proc=4,
    batch_size=100,
)

print(featurised_dataset)
```

You can see that we now have the extra columns for
* ``physicsml_atom_idxs``: The index of the atoms in the molecules
* ``physicsml_atom_numbers``: The atomic numbers mapped using the mapping dictionary
* ``physicsml_coordinates``: The coordinates of the atoms
* ``physicsml_total_atomic_energy``: The [total atomic self energy](../features/intro.html#atomic-energies)

The ``"as": "{feature_name}"`` kwarg controls how the computed feature names appear in the dataset. For more information,
see [Tweaking feature column names](https://exscientia.github.io/molflux/pages/datasets/featurising.html#tweaking-the-featurised-columns-names)
in the ``molflux`` docs.

## Splitting

Next, we need to split the dataset. For this, we use the simple ``shuffle_split`` (random split) with 80% training and
20% test. To split the dataset, we use the ``split_dataset`` function from ``molflux.datasets``.

```{code-cell} ipython3

from molflux.datasets import split_dataset
from molflux.splits import load_from_dict as load_split_from_dict

shuffle_strategy = load_split_from_dict(
    {
        "name": "shuffle_split",
        "presets": {
            "train_fraction": 0.8,
            "validation_fraction": 0.0,
            "test_fraction": 0.2,
        }
    }
)

split_featurised_dataset = next(split_dataset(featurised_dataset, shuffle_strategy))

print(split_featurised_dataset)
```

For more information about splitting datasets, see the ``molflux`` splitting [documentation](https://exscientia.github.io/molflux/pages/splits/intro.html).


## Training the model

We can now turn to training the model! ``physicsml`` models are accessed and used via ``molflux.modelzoo`` which defines the
standard API and handling. For more information, check out the ``molflux.modelzoo`` [docs](https://exscientia.github.io/molflux/pages/modelzoo/intro.html).

The recommended way to load models is by defining a model config (there are other ways, see the ``molflux.modelzoo`` [docs](https://exscientia.github.io/molflux/pages/modelzoo/intro.html)).
A model config is a dictionary with a ``"name"`` key (for the model name) and a ``"config"`` key (for the model config).
In the ``"config"``, we specify the ``x_features`` (the computed feature columns for the model), ``y_features`` (the properties
to fit), and a bunch of model specific kwargs. For a full description of model configs, see [``molflux`` layer](../structure/molflux_layer.html#model-loading).

In general, model configs can have defaults so that users do not need to specify them every time but here we show them
explicitly for illustration.


```{code-cell} ipython3
from molflux.modelzoo import load_from_dict as load_model_from_dict

model = load_model_from_dict(
    {
        "name": "ani_model",                        # model name
        "config": {
            "x_features": [                         # x features
                'physicsml_atom_idxs',
                'physicsml_atom_numbers',
                'physicsml_coordinates',
                'physicsml_total_atomic_energy',
            ],
            "y_features": ['u0'],                   # y features
            "which_ani": "ani2",                    # model specific kwarg to specify which ANI model to use (ani1 or ani2)
            "y_graph_scalars_loss_config": {        # the loss config for the y graph scalars
                "name": "MSELoss",
            },
            "optimizer": {                          # The optimizer config
                "name": "AdamW",
                "config": {
                    "lr": 1e-3,
                }
            },
            "datamodule": {                         # The datamodule config
                "y_graph_scalars": ['u0'],          # specify which y features are graph level scalars
                "pre_batch": "in_memory",           # pre batch the dataset for faster data loading
                "train": {"batch_size": 64},        # specify the training batch size
                "validation": {"batch_size": 128},  # specify the val batch size (which can be different from the train size)
            },
            "trainer": {                            # the trainer config
                "max_epochs": 10,                   # the maximum number of epochs
                "accelerator": "cpu",               # the accelerator, here cpu
                "logger": False,                    # whether to log losses
            }
        }
    }
)
```


Once loaded, we can simply train the model by calling the ``.train()`` method

```{code-cell} ipython3
model.train(
    train_data=split_featurised_dataset["train"]
)
```

Once trained, you can save the model by

```python
from molflux.core import save_model

save_model(model, "model_path", featurisation_metadata)
```

This will persist the model artefacts (the model weights checkpoint), the model config, the featurisation metadata, and
the requirements file of the environment the model was built in for reproducibility. For more on saving models, check out
the ``molflux`` [documentation](https://exscientia.github.io/molflux/pages/production/models.html).

After training, we can now compute some predictions and metrics! We load the ``regression`` suite of metrics which can
generate a variety of regression metrics and use the model predictions and the reference values to compute them.

```{code-cell} ipython3
import json

from molflux.metrics import load_suite
import matplotlib.pyplot as plt

preds = model.predict(
    split_featurised_dataset["test"],
    datamodule_config={"predict": {"batch_size": 256}}
)

regression_suite = load_suite("regression")

scores = regression_suite.compute(
    references=split_featurised_dataset["test"]["u0"],
    predictions=preds["ani_model::u0"],
)

print(json.dumps(scores, indent=4))

true_shifted = [x - e for x, e in zip(split_featurised_dataset["test"]["u0"], split_featurised_dataset["test"]["physicsml_total_atomic_energy"])]
pred_shifted = [x - e for x, e in zip(preds["ani_model::u0"], split_featurised_dataset["test"]["physicsml_total_atomic_energy"])]
plt.scatter(
    true_shifted,
    pred_shifted,
)
plt.plot([-0.1, 0.1], [-0.1, 0.1], c='r')
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()
```
