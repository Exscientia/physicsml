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

# QM datasets

The ``physicsml`` package uses the ``molflux.datasets`` module which provides efficient and convenient handling
of datasets and direct access to publicly available ones (such as ANI1ccx). It is built on top of HuggingFace [``datasets``](https://huggingface.co/docs/datasets/index).

As part of the ``molflux`` package, you can access multiple quantum mechanical datasets directly. The dataset files are
downloaded from source, processed on the fly, and the dataset is constructed and cached for later use. For more information
about ``molflux`` datasets, please see the [docs](https://exscientia.github.io/molflux/pages/datasets/intro.html). All of the datasets below require a kwarg specifying the
backend to process the molecules, either ``'rdkit'`` or ``'openeye'``

```python
from molflux.datasets import load_dataset

dataset = load_dataset("dataset_name", "backend_name")
```

## ``ani1x``

The ANI-1x and ANI-1ccx ML-based general-purpose datasets for organic molecules
were developed through active learning; an automated data diversification process.
The ANI-1x data set contains multiple quantum mechanical properties from 5M density
functional theory calculations, while the ANI-1ccx data set contains 500k data
points obtained with an accurate CCSD(T)/CBS extrapolation. Approximately 14 million
CPU core-hours were expended to generate this data. Multiple QM calculated properties
for the chemical elements C, H, N, and O are provided: energies, atomic forces, multipole
moments, atomic charges, etc.

```{note}
Description from [source](https://www.nature.com/articles/s41597-020-0473-z).

The molecules here are point clouds. They do not have any bonds.
```

## ``ani2x``

The new model, dubbed ANI-2x, is trained to three additional chemical elements: S, F, and Cl.
Additionally, ANI-2x underwent torsional refinement training to better predict molecular torsion
profiles. These new features open a wide range of new applications within organic chemistry and
drug development. These seven elements (H, C, N, O, F, Cl, and S) make up ~90% of drug-like molecules.

```{note}
Description from [source](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00121).

The molecules here are point clouds. They do not have any bonds.
```

## ``gdb9``

Computational de novo design of new drugs and materials requires rigorous and unbiased exploration of chemical compound
space. However, large uncharted territories persist due to its size scaling combinatorially with molecular size. We
report computed geometric, energetic, electronic, and thermodynamic properties for 134k stable small organic molecules
made up of CHONF. These molecules correspond to the subset of all 133,885 species with up to nine heavy atoms (CONF)
out of the GDB-17 chemical universe of 166 billion organic molecules. We report geometries minimal in energy, corresponding
harmonic frequencies, dipole moments, polarizabilities, along with energies, enthalpies, and free energies of atomization.
All properties were calculated at the B3LYP/6-31G(2df,p) level of quantum chemistry. Furthermore, for the predominant
stoichiometry, C7H10O2, there are 6,095 constitutional isomers among the 134k molecules. We report energies, enthalpies,
and free energies of atomization at the more accurate G4MP2 level of theory for all of them. As such, this data set
provides quantum chemical properties for a relevant, consistent, and comprehensive chemical space of small organic
molecules. This database may serve the benchmarking of existing methods, development of new methods, such as hybrid
quantum mechanics/machine learning, and systematic identification of structure-property relationships.

```{note}
Description from [source](https://www.nature.com/articles/sdata201422).
```

## ``spice``

SPICE dataset, a new quantum chemistry dataset for training potentials relevant to simulating drug-like small
molecules interacting with proteins. It contains over 1.1 million conformations for a diverse set of small
molecules, dimers, dipeptides, and solvated amino acids. It includes 15 elements, charged and uncharged molecules,
and a wide range of covalent and non-covalent interactions. It provides both forces and energies
calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory.

```{note}
Description from [source](https://github.com/openmm/spice-dataset).

The molecules here are point clouds. They do not have any bonds. The formal charges are added from the SMILES.
```

## ``pcqm4m_v2``

PCQM4Mv2 is a quantum chemistry dataset originally curated under the PubChemQC project.
Based on the PubChemQC, we define a meaningful ML task of predicting DFT-calculated HOMO-LUMO
energy gap of molecules given their 2D molecular graphs. The HOMO-LUMO gap is one of the most
practically-relevant quantum chemical properties of molecules since it is related to reactivity,
photoexcitation, and charge transport. Moreover, predicting the quantum chemical property only from
2D molecular graphs without their 3D equilibrium structures is also practically favorable. This is
because obtaining 3D equilibrium structures requires DFT-based geometry optimization, which is
expensive on its own.

We provide molecules as the SMILES strings, from which 2D molecule graphs (nodes are atoms and edges
are chemical bonds). We further provide the equilibrium 3D graph structure for training molecules.

```{note}
Description from [source](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/).
```
