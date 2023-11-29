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

In this tutorial we provide a simple example of training the ANI model on the QM9 dataset. We require the ``rdkit``
package, so make sure to ``pip install 'physicsml[rdkit]'`` to follow along!


## Loading the QM9 dataset

First, let's load the QM9 dataset and select 1000 random datapoints for this example.

```{code-cell} ipython3
import numpy as np
from molflux.datasets import load_dataset

dataset = load_dataset("gdb9", "rdkit")

print(dataset)

idxs = np.random.permutation(range(len(dataset)))
dataset = dataset.select(idxs[:1000])

print(dataset)
```

You can see that there is the ``mol_bytes`` column (which is the ``rdkit`` serialisation of the 3d molecules and the
remaining columns of computes properties.


## Featurising

Next, we will featurise the dataset. The ANI model requires only the atomic numbers, coordinates, and atomic self energies.

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


## Training the model

We can now turn to training the model! We choose the ``ani_model``.  To do so, we need to define the model config and
the ``x_features`` and the ``y_features``.

Once trained, we will get some predictions and compute some metrics!

```{code-cell} ipython3
import json

from molflux.modelzoo import load_from_dict as load_model_from_dict
from molflux.metrics import load_suite

import matplotlib.pyplot as plt

model = load_model_from_dict(
    {
        "name": "ani_model",
        "config": {
            "x_features": [
                'physicsml_atom_idxs',
                'physicsml_atom_numbers',
                'physicsml_coordinates',
                'physicsml_total_atomic_energy',
            ],
            "y_features": ['u0'],
            "which_ani": "ani2",
            "y_graph_scalars_loss_config": {
                "name": "MSELoss",
            },
            "optimizer": {
                "name": "AdamW",
                "config": {
                    "lr": 1e-3,
                }
            },
            "datamodule": {
                "y_graph_scalars": ['u0'],
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
)

model.train(
    train_data=split_featurised_dataset["train"]
)

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
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()
```
