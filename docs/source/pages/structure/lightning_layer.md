# Lightning layer

Training deep learning models involves a lot of boilerplate code. From loading the data, to batching, to writing training
loops, the overhead builds up quite quickly (not to mention the complexity of training on multiple devices/GPUs). Since
model training is such a crucial step, a lot of care needs to be given to all of these aspects to do it efficiently and robustly. This
requires a team of dedicated experts from machine learning practitioners to software engineers.

This is why we opted to choose ``lightning`` to handle all of the training code in ``physicsml``. ``lightning`` is a library that
provides a high level API for training deep learning models (using ``torch``) which combines both robustness, efficiency,
and complete flexibility to suit all sorts of applications. For more information, see [Lightning](https://lightning.ai/pytorch-lightning).

## Inner workings

In this section we briefly discuss the different aspects of lightning and how they are used in the ``physicsml`` package.
There are three main parts: ``modules``, ``datamodules``, and ``Trainers``.


### ``modules``

The lightning ``module`` contains all of the ``torch`` model code for the model to function. It has the familiar ``forward``
pass (and is a bona fide ``torch.nn.Module``). But, there is a lot more functionality built on top. It defines the ``training_step``
(and ``validation_step``) which are responsible for computing the loss of a batch passed though the model. ``modules``
also handle instantiating the optimizers and schedulers and can also perform logging. They also provide complete flexibility
to modify every part of the training loop via callbacks. For more information, see [Lightning Module](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)
and [Lightning callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html).

The ``physicsml`` package builds on top of this to provide a tailored module for 3d based models.

### ``datamodules``

The lightning ``datamodule`` is responsible for handling the data during training. It is essentially a wrapper around what is
usually the ``train_dataloader`` and the ``validation_dataloader`` to make the data handling more self-contained. For more
information, see [Lightning datamodule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).

### ``Trainers``

The lightning ``Trainer`` is the main class responsible for training. It uses both the ``module`` and the ``datamodule``
to run the training. It sets up the training using its specified config and relies on the methods defined in the ``module``
to run the training. For more information, see [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html).

## Configs

In the ``physicsml`` package, access to all of these objects is done via configs (which are validated via ``dataclasses``).
In this section, we go over the configs for the above components.

### ``module`` config

The ``module`` configs are specific to each model architecture. They specify the hyperparameters of the models. For more
information about the config for each model type, see [models](../models/intro.md).

### ``datamodule`` config

The ``datamodule`` config controls all aspects of dataloading. It takes in the follows kwargs

````{toggle}
* ``train: Dict[str, Any] = {"batch_size": 1}``

    Dictionary responsible for defining the ``train`` dataset's ``batch_size``.
* ``validation: Dict[str, Any] = {"batch_size": 1}``

    Dictionary responsible for defining the ``validation`` dataset's ``batch_size``.
* ``predict: Dict[str, Any] = {"batch_size": 1}``

    Dictionary responsible for defining the ``predict`` dataset's ``batch_size``.
* ``num_workers: Optional[str, int] = 0``

    The number of workers to use. Can set to ``"all"`` to use all workers. If running on CPU only machine make sure to
    set to 0 (otherwise processes can hang).
* ``num_elements: int = 0``

    The number of atomic elements used (for example 4 in ANI1x).
* ``y_node_scalars: Optional[List[str]] = None``

    The subset of ``y_features`` which are node level scalars (for example partial charges).
* ``y_node_vector: Optional[str] = None``

    The feature from``y_features`` which is a node level vector.
* ``y_edge_scalars: Optional[List[str]] = None``

    The subset of ``y_features`` which are edge level scalars.
* ``y_edge_vector: Optional[str] = None``

    The feature from``y_features`` which is a edge level vector.
* ``y_graph_scalars: Optional[List[str]] = None``

    The subset of ``y_features`` which are graph level scalars.
* ``y_graph_vector: Optional[str] = None``

    The feature from``y_features`` which is a graph level vector (for example forces).
* ``cut_off: float = 5.0``

    The cut-off for determining the neighbourhoods.
* ``pbc: Optional[Tuple[bool, bool, bool]] = None``

    Whether to use periodic boundary conditions.
* ``cell: Optional[List[List[float]]] = None``

    The dimensions of the unit cell for periodic boundary conditions.
* ``self_interaction: bool = False``

    Whether to include self connections (i.e. edges from an atom to itself).
* ``pre_batch: Optional[Literal["in_memory", "on_disk"]] = None``

    Pre-batching method. Speeds up dataloading and allows for training with minimal CPUs.
    Can be pre batching in memory (for datasets up to 1M datapoints) or on disk (for larger datasets).
````

### ``Trainer`` config

The ``Trainer`` config controls all aspects of training. It is defined in the Lightning [docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api)
but we show the most useful kwargs here again for convenience

````{toggle}
* ``accelerator: str = "auto"``

    The accelerator or device to use for training (``"cpu"``, ``"gpu"``, etc...)
* ``devices: Union[List[int], str, int] = "auto"``

    The number or list of devices to use.
* ``strategy: Union[str, Dict[str, Any]] = "auto"``

    The strategy to use (``"auto"``, ``"ddp"``, etc..)
* ``callbacks: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]], None] = None``

    The callbacks to use. A list of dictionaries specifying the name and config of each callback.
* ``default_root_dir: Optional[str] = "training"``

    The dir in which to save the logs and checkpoints.
* ``enable_checkpointing: bool = False``

    Whether to enable checkpointing or not.
* ``max_epochs: Optional[int] = 1``

    The maximum number of epochs to run for.
* ``min_epochs: Optional[int] = None``

    The minimum number of epochs to run for.
* ``precision: Union[int, str] = 32``

    The precision to use (32 or 64).
* ``gradient_clip_algorithm: Optional[str] = None``

    The gradient clipping algorithm to use (``norm`` or ``value``).

* ``gradient_clip_val: Optional[Union[int, float]] = None``

    The value to clip at.

````

### Restarting training from a checkpoint

The ``lightning`` ``Trainer`` provides a way to continue training from a saved checkpoint. We surface this at the ``train``
method of the model since it used in the ``Trainer.fit`` method (and not at instantiation)

```python
model.train(
  train_data=train_dataset,
  validation_data=validation_dataset,
  ckpt_path="path_to_ckpt",
)
```
