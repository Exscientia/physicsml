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

# ASE

One of the functionalities that ``physicsml`` provides is an ASE Calculator. This allows you to plug into the ASE ecosystem
easily.

```{note}
To use the ASE functionality, you need to ``pip install "physicsml[ase]"``.
```

Let’s look at an example and discuss the available kwargs in detail. Loading a physicsml ASE calculator is simple.
You can just point at a model path and get a calculator. First, let's quickly train a model

```{code-cell} ipython3
:tags: [hide-cell]

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

model_config =     {
    "name": "mace_model",
    "config": {
        "x_features": [
            'physicsml_atom_idxs',
            'physicsml_atom_numbers',
            'physicsml_coordinates',
            'physicsml_total_atomic_energy',
        ],
        "y_features": [
            'u0',
        ],
        "num_node_feats": 5,
        "max_ell": 2,
        "hidden_irreps": "12x0e + 12x1o",
        "correlation": 2,
        "y_graph_scalars_loss_config": {
            "name": "MSELoss",
        },
        "optimizer": {
            "name": "AdamW",
            "config": {
                "lr": 1e-3,
            }
        },
        "scheduler": None,
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
        }
    }
}

from molflux.modelzoo import load_from_dict
from molflux.core import save_model

model = load_from_dict(model_config)
model.module = model._instantiate_module()

save_model(model, "trained_mace_model", featurisation_metadata)
```

Now, we can load the ASE calculator by doing

```{code-cell} ipython3
from physicsml.plugins.ase.load import to_ase_calculator

ase_calculator = to_ase_calculator(model_path="trained_mace_model")
```

The ``to_ase_calculator`` method can take in the following kwargs

```{toggle}
* ``model_path: str``

    The path to the trained ``physicsml`` model.
* ``precision: Literal["32", "64"] = "64"``

    The precision to use.
* ``position_scaling: float = 1.0``

    The position scaling to use.
* ``output_scaling: float = 1.0``

    The output scaling to use.
* ``device: Literal["cpu", "cuda"] = "cpu"``

    The device to use.
```

For example, we can use the ASE calculator to get some energy and forces

```{code-cell} ipython3
import numpy as np
from ase import Atoms

# Define ASE atom object
atoms = Atoms("H2O", positions=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]))
atoms.set_calculator(ase_calculator)

print(atoms.get_potential_energy())
print(atoms.get_forces())
```
