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

# Transfer learning

In this tutorial, we provide an example of using the transfer learning functionality. We require the ``rdkit``
package, so make sure to ``pip install 'physicsml[rdkit]'`` to follow along!

We will train an EGNN model on the ``lumo`` energy of QM9 and then transfer this model to predict the
``u0`` energy.

## Pre-trained model

First, let's load a truncated QM9 dataset with 1000 datapoints

```{code-cell} ipython3
import numpy as np
from molflux.datasets import load_dataset_from_store

dataset = load_dataset_from_store("gdb9_trunc.parquet")

print(dataset)

idxs = np.random.permutation(range(len(dataset)))
dataset = dataset.select(idxs[:1000])

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

You can see that there is the ``mol_bytes`` column (which is the ``rdkit`` serialisation of the 3d molecules) and the
remaining columns of computes properties.

Next, we will featurise the dataset. In this example, we start by using the atomic numbers only (since we do not require
self energies for the ``lumo`` energy).

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

We can now turn to pre training the model! To do so, we need to define the model config and the ``x_features`` and
the ``y_features``. Once trained, we will save it to be used for transferring.

```{code-cell} ipython3
import json

from molflux.modelzoo import load_from_dict as load_model_from_dict
from molflux.core import save_model

model = load_model_from_dict(
    {
        "name": "egnn_model",
        "config": {
            "x_features": [
                'physicsml_atom_idxs',
                'physicsml_atom_numbers',
                'physicsml_coordinates',
            ],
            "y_features": ['lumo'],
            "num_node_feats": 5,
            "num_layers": 2,
            "c_hidden": 12,
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
                "y_graph_scalars": ['lumo'],
                "num_elements": 5,
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
)

model.train(
    train_data=split_featurised_dataset["train"]
)

save_model(model, "pre_trained_model", featurisation_metadata)
```

We now have a dummy pre-trained model.


## Transfer learning

Finally, we come to the transfer learning. First, we need to re-featurise the dataset to include the atomic self energies
for ``'u0'`` and then split it for training (ignoring that this is the same dataset for pretraining and transferring, it's only
for demonstration).

```{code-cell} ipython3
import logging
logging.disable(logging.CRITICAL)

from molflux.core import featurise_dataset
from molflux.datasets import split_dataset
from molflux.splits import load_from_dict as load_split_from_dict

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

Now we need to define the model config. This will be exactly the same as the pre-trained model config with ``lumo``
substituted with ``u0`` and with the addition of the transfer learning config. For more information about the
transfer learning config, see [here](../structure/transfer_learning.md).


```{code-cell} ipython3
model_config = {
    "name": "egnn_model",
    "config": {
        "x_features": [
            'physicsml_atom_idxs',
            'physicsml_atom_numbers',
            'physicsml_coordinates',
            'physicsml_total_atomic_energy',
        ],
        "y_features": ['u0'],
        "num_node_feats": 5,
        "num_layers": 2,
        "c_hidden": 12,
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
            "num_elements": 5,
            "cut_off": 5.0,
            "pre_batch": "in_memory",
            "train": {"batch_size": 64},
            "validation": {"batch_size": 128},
        },
        "trainer": {
            "max_epochs": 10,
            "accelerator": "cpu",
            "logger": False,
        },
        "transfer_learning": {
            "pre_trained_model_path": "pre_trained_model",
            "modules_to_match": {
                "egnn": "egnn",
            },
            "stages": [
                {
                    "freeze_modules": ["egnn"],
                    "datamodule": {
                        "train": {"batch_size": 128},
                    },
                    "optimizer": {
                        "config": {
                            "lr": 1e-2
                        }
                    }
                },
                {
                    "trainer": {
                        "max_epochs": 4,
                    },
                    "optimizer": {
                        "config": {
                            "lr": 1e-4
                        }
                    }
                }
            ]
        }
    }
}

```

As you can see, we first specify the path to the pre-trained model. The EGNN model contains two main submodules: the
``egnn`` backbone (message passing) and the ``pooling_head`` (for pooling and generating predictions). In this
example, we choose to match only the backbone (since the ``lumo`` and ``u0`` tasks are different and the ``pooling_head``
will not contain any useful learnt information).

Next, we specify a two stage transfer learning. In the first, we choose to freeze the ``egnn`` backbone and to train the
``pooling_head``. Additionally, we override some kwargs (such as batch size and learning rate). Notice that you
only need to specify the kwargs you want to override (such as learning rate) and all the rest will be used from the
main config. In the second stage we train the whole model (no frozen modules) at a lower learning rate for less epochs.


So let's run the training!

```{code-cell} ipython3
import logging
logging.disable(logging.NOTSET)

from molflux.modelzoo import load_from_dict as load_model_from_dict
model = load_model_from_dict(model_config)

model.train(
    train_data=split_featurised_dataset["train"]
)
```

You can see the logs from the transfer learning above. First, the ``egnn`` module is matched. Then the first stage
starts: freezes the ``egnn`` module (you can see the number of trainable parameters) and trains for 10 epochs. Then the
second stage starts and trains all the parameters for 4 epochs.

And that's it!
