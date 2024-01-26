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

# Energy and forces training

In this tutorial we provide an example of training ``physicsml`` models on both energies and forces from the ``ani1x``
dataset. In this example we choose the ``nequip_model``. We require the ``rdkit``package, so make sure to
``pip install 'physicsml[rdkit]'`` to follow along!


## Loading the ANI1x dataset

First, let's load the ``ani1x`` dataset. We will load a truncated version of the dataset (as it's too large to load
in the docs). For more information on the loading and using dataset, see the ``molflux`` [documentation](https://exscientia.github.io/molflux/pages/datasets/basic_usage.html).

```{code-cell} ipython3
from molflux.datasets import load_dataset_from_store

dataset = load_dataset_from_store("ani1x_truncated.parquet")

print(dataset)
```

````{note}
The dataset above is a truncated version to run efficiently in the docs. For running this locally, load the entire dataset
by doing

```{code-block} python
from molflux.datasets import load_dataset

dataset = load_dataset("ani1x", "rdkit")
```
````

You can see that there is the ``mol_bytes`` column (which is the ``rdkit`` serialisation of the 3d molecules) and the
remaining columns of computes properties.


## Featurising

Next, we will featurise the dataset. We extract the atomic numbers, coordinates, and atomic self energies. For more
information on the physicsml features, see [here](../features/intro.md).

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
                        },
                        "atomic_energies": {
                            1: -0.5894385,
                            6: -38.103158,
                            7: -54.724035,
                            8: -75.196441,
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
* ``physicsml_total_atomic_energy``: The total atomic self energy

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

We can now turn to training the model! We choose the ``nequip_model`` (or actually a smaller version of it for the example).
To do so, we need to define the model config.

```{code-cell} ipython3
model_config =     {
    "name": "nequip_model",
    "config": {
        "x_features": [
            'physicsml_atom_idxs',
            'physicsml_atom_numbers',
            'physicsml_coordinates',
            'physicsml_total_atomic_energy',
        ],
        "y_features": [
            'wb97x_dz.energy',
            'wb97x_dz.forces',
        ],
        "num_node_feats": 4,
        "num_features": 5,
        "num_layers": 2,
        "max_ell": 1,
        "compute_forces": True,
        "y_graph_scalars_loss_config": {
            "name": "MSELoss",
            "weight": 1.0,
        },
        "y_node_vector_loss_config": {
            "name": "MSELoss",
            "weight": 0.5,
        },
        "optimizer": {
            "name": "AdamW",
            "config": {
                "lr": 1e-3,
            }
        },
        "scheduler": None,
        "datamodule": {
            "y_graph_scalars": ['wb97x_dz.energy'],
            "y_node_vector": 'wb97x_dz.forces',
            "num_elements": 4,
            "cut_off": 5.0,
            "pre_batch": "in_memory",
            "train": {"batch_size": 64},
            "validation": {"batch_size": 128},
        },
        "trainer": {
            "max_epochs": 10,
            "accelerator": "cpu",
            "logger": False,
        }
    }
}
```

In the ``y_features`` we specify all the columns required for training (energy and forces). In the ``datamodule``, we provide
more details about what each y feature is: energy is a ``y_graph_scalars`` and forces are a ``y_node_vector``. We also
specify that the model should ``compute_forces`` by computing the gradients of the ``y_graph_scalars`` instead of predicting
a ``y_node_vector`` directly (which is possible in some models). Finally, we specify two loss configs, one for the
``y_graph_scalars_loss_config`` and one for ``y_node_vector_loss_config`` (each of which can be weighted with a ``weight``).

We can now train the model and compute some predictions!

```{code-cell} ipython3
import json

from molflux.modelzoo import load_from_dict as load_model_from_dict
from molflux.metrics import load_suite

import matplotlib.pyplot as plt

model = load_model_from_dict(model_config)

model.train(
    train_data=split_featurised_dataset["train"]
)

preds = model.predict(
    split_featurised_dataset["test"],
    datamodule_config={"predict": {"batch_size": 256}}
)

print(preds.keys())
```

As you can see the predictions include an energy prediction and a forces prediction. Finally, we compute some metrics

```{code-cell} ipython3
regression_suite = load_suite("regression")

scores = regression_suite.compute(
    references=split_featurised_dataset["test"]["wb97x_dz.energy"],
    predictions=preds["nequip_model::wb97x_dz.energy"],
)

print(json.dumps(scores, indent=4))

true_shifted = [x - e for x, e in zip(split_featurised_dataset["test"]["wb97x_dz.energy"], split_featurised_dataset["test"]["physicsml_total_atomic_energy"])]
pred_shifted = [x - e for x, e in zip(preds["nequip_model::wb97x_dz.energy"], split_featurised_dataset["test"]["physicsml_total_atomic_energy"])]
plt.scatter(
    true_shifted,
    pred_shifted,
)
plt.plot([-0.3, 0.3], [-0.3, 0.3], c='r')
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()
```
