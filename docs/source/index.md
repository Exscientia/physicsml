PhysicsML
=======

**PhysicsML** is a package for all physics based/related models. It provides a standardised interface for handling,
building, and training 3d models!

````{grid} 2
```{grid-item-card} 💡[Tutorials](pages/tutorials/gdb9_training.md)
Learn the basics of PhysicsML by jumping straight into examples!
```
```{grid-item-card} 📏 [Standard API](pages/philosophy/philosophy.md)
The Standard API of PhysicsML allows you to easily use many different models. Learn more here!
```
````

To learn more about the individual components of PhysicsML, check out the documentation for each module below!

````{grid} 3
```{grid-item-card} 🗂 [Datasets](pages/datasets/qm_datasets.md)
Learn more about the 3d datasets provided!
```
```{grid-item-card} 🤖 [PhysicsML models](pages/models/intro.md)
Jump into the many model architectures accessible through PhysicsML!
```
```{grid-item-card} 📈 [Plugins](pages/plugins/openmm.md)
PhysicsML provides plugins to popular molecular dynamics engines.
```
````

```{toctree}
---
hidden:
glob:
maxdepth: 2
caption: Philosophy
---
pages/philosophy/philosophy
pages/philosophy/molflux
```

```{toctree}
---
hidden:
glob:
maxdepth: 2
caption: Tutorials
---
pages/tutorials/gdb9_training
pages/tutorials/ani1x_energy_forces_training
pages/tutorials/gdb9_uncertainty
pages/tutorials/transfer_learning
```

```{toctree}
---
hidden:
glob:
maxdepth: 2
caption: PhysicsML structure
---
pages/structure/physicsml_structure
pages/structure/molflux_layer
pages/structure/lightning_layer
pages/structure/torch_layer
pages/structure/transfer_learning
```

```{toctree}
---
hidden:
glob:
maxdepth: 2
caption: Datasets
---
pages/datasets/qm_datasets
```

```{toctree}
---
hidden:
glob:
maxdepth: 2
caption: Features
---
pages/features/intro
pages/features/how_to_add_reps
```

```{toctree}
---
hidden:
glob:
maxdepth: 2
caption: Models
---
pages/models/intro
pages/models/allegro
pages/models/ani
pages/models/egnn
pages/models/mace
pages/models/nequip
pages/models/tensor_net
pages/models/how_to_add_models
```

```{toctree}
---
hidden:
glob:
maxdepth: 2
caption: Plugins
---
pages/plugins/openmm
pages/plugins/ase
```

```{toctree}
---
hidden:
glob:
maxdepth: 2
caption: Reference
---
pages/reference/catalogue/index
API <pages/reference/api/modules.rst>
```
