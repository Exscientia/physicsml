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

# QM9 with uncertainty

In this tutorial we provide an example of training a model with uncertainty on the QM9 dataset. To do this, we will use a
model that supports uncertainty prediction, ``ensemble_ani_model``. For more information on which models support uncertainty,
check out the [models page](../models/intro.md). We require the ``rdkit`` package, so make sure to ``pip install 'physicsml[rdkit]'``
to follow along!


## Loading the QM9 dataset

First, let's load a truncated QM9 dataset with 1000 datapoints. For more information on the loading
and using dataset, see the ``molflux`` [documentation](https://exscientia.github.io/molflux/pages/datasets/basic_usage.html).

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

You can see that there is the ``mol_bytes`` column (which is the ``rdkit`` serialisation of the 3d molecules and the
remaining columns of computes properties.


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

We can now turn to training the model! The ``ensemble_ani_model`` is composed of a single AEV computer and a number of
neural network heads. These heads are trained from different randomly initialised parameters with the idea that each one
converges to a different minimum of the loss landscape. The final prediction is the mean of the individual predictions
with a standard deviation computed from their variance. The idea is that if all the models produce a similar prediction for
a datapoint then it must be "more certain", whereas if the predictions are different then the uncertainty is higher.

We start by specifying the model config and the ``x_features``, the ``y_features``, and the ``n_models`` the number of
heads to use.

```{code-cell} ipython3
from molflux.modelzoo import load_from_dict as load_model_from_dict

model = load_model_from_dict(
    {
        "name": "ensemble_ani_model",
        "config": {
            "x_features": [
                'physicsml_atom_idxs',
                'physicsml_atom_numbers',
                'physicsml_coordinates',
                'physicsml_total_atomic_energy',
            ],
            "y_features": ['u0'],
            "which_ani": "ani2",
            "n_models": 3,
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
                "train": {"batch_size": 256},
                "validation": {"batch_size": 128},
            },
            "trainer": {
                "max_epochs": 20,
                "accelerator": "cpu",
                "logger": False,
            }
        }
    }
)

model.train(
    train_data=split_featurised_dataset["train"]
)
```

Now that the model is trained, we can inference it to get some predictions! Apart from the usual ``predict`` method (which
returns the energy predictions), the uncertainty models support ``predict_with_std`` which returns a tuple of energy
predictions and their corresponding standard deviation predictions. For more information about the uncertainty API in
``physicsml`` models, see the ``molflux`` [documentation](https://exscientia.github.io/molflux/pages/modelzoo/uncertainty.html) on which it is based.

Below we demonstrate how to get predictions and standard deviations and plot them!

```{code-cell} ipython3
import json
import matplotlib.pyplot as plt

from molflux.metrics import load_suite


preds, stds = model.predict_with_std(
    split_featurised_dataset["test"],
    datamodule_config={"predict": {"batch_size": 256}}
)

regression_suite = load_suite("regression")

scores = regression_suite.compute(
    references=split_featurised_dataset["test"]["u0"],
    predictions=preds["ensemble_ani_model::u0"],
)

print(json.dumps(scores, indent=4))

true_shifted = [x - e for x, e in zip(split_featurised_dataset["test"]["u0"], split_featurised_dataset["test"]["physicsml_total_atomic_energy"])]
pred_shifted = [x - e for x, e in zip(preds["ensemble_ani_model::u0"], split_featurised_dataset["test"]["physicsml_total_atomic_energy"])]

plt.errorbar(
    true_shifted,
    pred_shifted,
    yerr=stds["ensemble_ani_model::u0::std"],
    fmt='o',
)
plt.plot([-0.1, 0.1], [-0.1, 0.1], c='r')
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()
```
