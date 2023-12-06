# Transfer learning

One functionality that ``physicsml`` inherits from ``molflux[lightning]`` is the ability to do transfer learning. It has been
empirically shown that in many cases training a model on a large diverse dataset and then fine tuning it on a small dataset
leads to better performance. At its core, transfer learning improves performance by effectively providing better weight initialisations
for models to begin training from.

To help you perform transfer learning smoothly, we have created a config driven transfer learning functionality. In this
section, we go through the config template and explain the functions of each part.

For a tutorial on transfer learning, see [Transfer learning tutorial](../tutorials/transfer_learning.md).

## Transfer learning config

The transfer learning config has the following form

* ``pre_trained_model_path``: Specifies the path to a pre-trained model.
* ``modules_to_match``: Specified how to match the weights.
* ``stages``: Specifies how the transfer learning stages are run.


### ``pre_trained_model_path``

This must be a path to a pre-trained ``physicsml`` model. Make sure that this points to a model with the same architecture
and hyperparameters as the model you are training. If it is not, then matching the weights will fail.

### ``modules_to_match``

This is a dictionary specifying how to match the weights from the pre-trained model to the new model. The dictionary has
the following format

```python
{
    "module_name_in_pre_trained_model": "module_name_in_new_model",
}
```

You can specify any of the names of the child modules as well (in the format ``module.submodule.subsubmodule``). Any
child modules of the higher-level modules you specify will also be matched. If you do not specify this dictionary (which is
defaulted to ``None``), then all modules will be matched.

If the weights cannot be matched (because of a wrong module name or wrong parameters shape), then it will fail with an
error specifying where it failed.

### ``stages``

Finally, the ``stages`` specifies the structure of the transfer learning. In general, transfer learning is
done in consecutive stages. Each stage is essentially a separate training run which reserves its final model weights for the
next stage. In each of these stages, you can override any of the training kwargs for the ``trainer``, ``datamodule``,
``optimizer``, and ``scheduler``. For each of these, any specified kwargs will be overridden and any unspecified kwargs will
be taken from the initial definitions of these configs. This allows you to control how the stages run. Importantly, you
can also specify a list of module names in ``freeze_modules`` whose weights you would like to freeze in that stage.

The complete ``stage`` config looks like

```python
{
    "freeze_modules": [...],
    "trainer": {},
    "datamodule": {},
    "optimizer": {},
    "scheduler": {},
}
```
