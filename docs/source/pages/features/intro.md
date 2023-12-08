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

# PhysicsML features

The ``physicsml`` package provides its own featuriser to extract relevant features for 3d models. The featurisers are
built on top of the ``molflux.features`` (see [docs](https://exscientia.github.io/molflux/pages/features/intro.html)
for more info).

The featuriser is built on top of either molecules from ``rdkit`` and ``openeye``. It takes in molecule objects or
their binary serialisation and uses the built-in ``rdkit`` or ``openeye`` functions to extract a bunch of different atom
and bond features. By default, it will also extract the coordinates of the molecules.

## Config template

First, let's take a look at the featurisation config template. This is based on the ``molflux`` config
(see [here](https://exscientia.github.io/molflux/pages/production/featurisation.html)).

```python
featurisation_metadata = {
    "version": 1,
    "config": [
        {
            "column": <molecule_column>,
            "representations": [
                {
                    "name": "physicsml_features",
                    "config": {
                        "atomic_number_mapping": <mapping dict>,
                        "atomic_energies": <energies dict>,
                        "atom_features": <list of atom features>,
                        "bond_features": <list of bond features>,
                        "backend": <the backend name>,
                    },
                    "as": "{feature_name}",
                }
            ]
        }
    ]
}
```

The featuriser is called ``physicsml_features``. To featurise the QM9 dataset for example, you can do

```{code-cell} ipython3
import logging
logging.disable(logging.CRITICAL)

from molflux.datasets import load_dataset_from_store
from molflux.core import featurise_dataset

dataset = load_dataset_from_store("gdb9_trunc.parquet")

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

You can see the additional feature columns added to the dataset. We discuss each one (and some more) below.


### ``atomic_number_mapping``

The only required kwarg in the config is the ``atomic_number_mapping``. This defines how the atomic numbers are mapped
into indices for one-hot encoding. For example, in the ``ani1x`` dataset, this would look like ``{1: 0, 6: 1, 7: 2, 8: 3}``.
The computed features will be a list of mapped atomic numbers for each molecule, the list of the indices of the atoms
in the molecule, and the coordinates of the atoms. For example, CH4 would be

```python
{
    'physicsml_atom_numbers': [1, 0, 0, 0, 0],
    'physicsml_atom_idxs': [0, 1, 2, 3, 4],
    'physicsml_coordinates': [
        [...],  # C
        [...],  # H
        [...],  # H
        [...],  # H
        [...],  # H
    ]
}
```

### ``atomic_energies``

This is used for adding the self energies of molecules in energy prediction models and is vital for achieving good performance.
It essentially transforms the problem from predicting total energies to predicting interaction energies. The
config can be specified simply as ``{atomic_num: energy}``. For example with the ``ani1x`` dataset, we have

```python
{
    1: -0.60198056,
    6: -38.0774958,
    7: -54.7522543,
    8: -75.2252160,
}
```

The computed feature would be a float (the sum of all the self energies in a molecule). For example, CH4 would be

```python
{
    'physicsml_total_atomic_energy': -40.48541804,
}
```

```{important}
Make sure the energies are in the same unit as the total energy!
```

Since charged atoms have different self energies than their neutral counterparts, the featuriser also supports providing
energies of charged species. This must be specified in a nested dictionary as ``{atomic_num: {formal_charge: energy}}``.
If not specified, the featuriser will assume that the energies provided are for the neutral atoms. The atomic-number-charge-energy
dict can be provided for only some atomic numbers while others can just be specified as floats.

```{note}
If only a single atomic energy (float) is specified and if the featuriser encounters a charged atom, it will raise a
warning and use the only specified energy for that atom. If the value is a dictionary of {charges: atomic energies} and
the featuriser encounters a charged atom which is not specified in the dictionary, then it will raise a ``RuntimeError``
and fail.
```

In general, there are two ways to obtain atomic self energies for featurisation: From numerical QM solutions or via dataset
least squares regression fitting. For the second option, we provide a convenience function to do so. It uses the
molecule column, the energy column, and the backend to perform a least squares regression using the number and types of
atoms in the molecules against the energy

```{code-cell} opython3
from physicsml.utils import get_atomic_energies

get_atomic_energies(
    dataset["mol_bytes"],
    dataset['u0'],
    backend="rdkit"
)
```


### ``atom_features``

Apart from atomic numbers, we optionally provide a way to compute atomic features which will be concatenated to the node features
later on in the dataloader. There are a few to choose from:

* ``'formal_charge'``: The formal charge of the atom
* ``'degree'``:  The degree of the atom
* ``'stereo'``:  The stereochemistry of the atom
* ``'is_in_ring'``:  Whether the atom is in a ring
* ``'is_aromatic'``:  Whether the atom is in an aromatic ring
* ``'is_chiral'``: Whether the atom is chiral

You can specify any or all of these features as a list.

```{note}
You do not have to worry about the order of the strings in the list, the featuriser will order them alphabetically for you.
```

The computed features would be a list of list (one for each atom in the molecule) with the concatenated one-hot encoding of
the features. For CH4, it would be

```python
{
    'physicsml_atom_features': [
        [...],  # C features
        [...],  # H features
        [...],  # H features
        [...],  # H features
        [...],  # H features
    ]
}
```

```{warning}
We do not guarantee that the molecules being featurised have correct, sanitised atom features. That is the user's responsibility.
```


### ``bond_features``

We also optionally provide a way to compute bond features which can be concatenated to edge features later on in the
models. There are a few to choose from:

* ``'order'``: The order of the bond
* ``'stereo'``: The stereochemistry of the bond
* ``'is_in_ring'``: Whether the bond is in a ring
* ``'is_chiral'``: Whether the bond is chiral
* ``'is_rotatable'``: Whether the bond is rotatable

You can specify any or all of these features as a list.

```{note}
You do not have to worry about the order of the strings in the list, the featuriser will order them alphabetically for you.
```

The computed features would be sparsely encoded as a list of list (one for each bond in the molecule) with the concatenated
one-hot encoding of the features and the list of bond indices. For CH4, it would be

```python
{
    'physicsml_bond_idxs': [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ],
    'physicsml_bond_features': [
        [...],  # C-H (0, 1) features
        [...],  # C-H (0, 2) features
        [...],  # C-H (0, 3) features
        [...],  # C-H (0, 4) features
    ]
}
```

```{warning}
We do not guarantee that the molecules being featurised have correct, sanitised bond features. That is the user's responsibility.
```

```{important}
If ``bond_features`` are computed and specified in the model config, then edge features (notice, edge not bond) will be created
for all the edges in the input graph. The edges which are bona fide bonds will use the features computed here. The edges which
are not bonds will use a feature vector of zeros with the same dimension as the ``bond_features``.
```

### ``backend``

We provide multiple backends for computing and extracting features. Currently, we support ``rdkit`` (public, free)
and ``openeye`` (proprietary, paid).

```{warning}
Make sure that you load the datasets and featurise them with the same backend!
```
