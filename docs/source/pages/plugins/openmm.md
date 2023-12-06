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

# OpenMM

The ``physicsml`` package provides an OpenMM integration via the ``openmm-ml`` package. This integration allows for standardised
access and use of all ``physicsml`` models in OpenMM.

```{note}
To use the OpenMM functionality, you need to create an environment with ``openmm-ml`` and ``pip install "physicsml[openmm]"`` in it.
```

## PhysicsML ``MLPotential``

The ``physicsml`` package provides an integration into the ``MLPotential`` class of ``openmm-ml``. The class takes the
following kwargs

```{toggle}
* ```"physicsml_model"```,

    The ``physicsml`` integration name.
* ``model_path: str``

    The path to the trained ``physicsml`` model.
* ``precision: Literal["32", "64"] = "64"``

    The precision to use. OpenMM usually uses 64.
* ``position_scaling: float = 1.0``

    The position scaling to use. OpenMM often uses nanometers, whereas models are often trained in Angstroms, so a
    scaling of ``10.0`` must be applied.
* ``output_scaling: float = 1.0``

    The output scaling to use. OpenMM often uses kJ/mol, whereas models are trained on kcal/mol, so a scaling of 4.184
    must be applied.
* ``device: Literal["cpu", "cuda"] = "cpu"``

    The device to use.
```

## Example

Here, we present a brief example of using a trained MACE model in the OpenMM integration.

First, we train a model. For the purposes of this tutorial, we just load, initialise the parameters randomly and save the model.

```{note}
Note that all results below are not physical and are presented for illustration purposes only. For more info about
training models, checkout the [tutorials](../tutorials/gdb9_training.md).
```

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

Now, we can use this trained model (which is saved in ``trained_mace_model``) in the OpenMM integration. We use the
alanine dipeptide system in vacuum.

```{code-block} ipython3
import openmm as mm
import openmm.app as app
from openmmml.mlpotential import MLPotential
from physicsml.plugins.openmm.physicsml_potential import PhysicsMLPotentialImplFactory

# You can download the pdb file from https://github.com/openmm/openmm-ml/blob/main/test/alanine-dipeptide-explicit.pdb.
# Here, we truncate it to remove all the water molecules for speed
pdb = app.PDBFile("alanine-dipeptide-truncated.pdb")

# specify the Mace potential
potential = MLPotential(
    "physicsml_model",
    model_path="trained_mace_model",
    precision="64",
    position_scaling=10.0,
    output_scaling=4.184 * 627,
    device="cpu",
)

# set the platform to run on
platform_openmm = mm.Platform.getPlatformByName("CPU")

mm_system = potential.createSystem(pdb.topology)

# create the context for mm system
mm_context = mm.Context(mm_system, mm.VerletIntegrator(0.001), platform_openmm)

# set the positions
mm_context.setPositions(pdb.positions)

# get energy
energy = mm_context.getState(getEnergy=True).getPotentialEnergy()

print(energy)
```
