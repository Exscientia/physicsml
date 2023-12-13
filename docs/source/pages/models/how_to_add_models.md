# Add your own model architecture

While ``physicsml`` ships with a large catalogue of available model-architectures, you can also add you own
state-of-the-art architecture and make it more widely available to all users of the package. In this guide,
we will provide an overview of how to do this.

There are three main components to adding a new model: The model config, ``molflux`` model API handler, and ``torch`` module
code.


## The model config

The most basic part of adding a new model is defining the model config. The general model config looks like this:

```python
from typing import Dict, Literal, Optional

from pydantic.dataclasses import dataclass

from physicsml.lightning.config import ConfigDict, PhysicsMLModelConfig


@dataclass(config=ConfigDict)
class MyModelConfig(PhysicsMLModelConfig):
    # all model kwargs
    ...
```

The config is a pydantic dataclass which takes care of validating the inputs. It also inherits the ``PhysicsMLModelConfig``
which provides the shared kwargs (such as the ``datamodule`` config and the ``optimizer`` config). In here, you can specify
whatever your model requires for training and inference. A good simple example to follow is the [EGNN model config](https://github.com/Exscientia/physicsml/blob/main/src/physicsml/models/egnn/supervised/default_configs.py).

## The ``molflux`` model

The next part is the ``molflux`` model wrapper. This is responsible for handling all the training, inferencing, loading,
and saving functionality. If your model is a generic GNN, then this a simple wrapper class

```python
from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.my_model.supervised.default_configs import MyModelConfig

from physicsml.models.my_model.supervised.my_model_module import MyModelModule


class MyModel(PhysicsMLModelBase[MyModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="my model description",
            config_description="config description",
        )

    @property
    def _config_builder(self) -> Type[EGNNModelConfig]:
        return MyModelConfig

    def _instantiate_module(self) -> Any:
        return MyModelModule(
            model_config=self.model_config,
        )
```

The class inherits the ``PhysicsMLModelBase[MyModelConfig]`` class. The config in the square brackets is an inherited generic
for typing purposes.

You need to specify the ``ModelInfo`` (which has a description of the model and the config), the ``_config_builder`` which
returns the config class (for instantiation internally), and the ``_instantiate_module`` which returns an initialised ``lightning``
module (more on that below). For a simple example of this, check out the [EGNN model](https://github.com/Exscientia/physicsml/blob/main/src/physicsml/models/egnn/supervised/egnn_model.py).

If your model requires a specialised dataset and dataloader, then you can override the ``_datamodule_builder`` which
returns specific datamodule class for your model. For an example of this, see the [ANI model](https://github.com/Exscientia/physicsml/blob/main/src/physicsml/models/ani/supervised/ani_model.py).

```{note}
Uncertainty models need to inherit the ``PhysicsMLUncertaintyModelBase`` from ``physicsml.lightning.model_uncertainty``.
This class handles the additional API functionality for uncertainty models. For an example, see the
[MeanVarEGNNModel](https://github.com/Exscientia/physicsml/blob/main/src/physicsml/models/egnn/mean_var/mean_var_egnn_model.py).
```

### Make your model discoverable

To make your model discoverable in ``physicsml`` and ``molflux``, you need to register it as a plugin. You can do that in
the ``pyproject.toml`` file of your repo and under ``[project.entry-points.'molflux.modelzoo.plugins.physicsml']``,
add a plugin to your model class as follows

```{code-block} ini
[project.entry-points.'molflux.modelzoo.plugins.physicsml']
name_of_model = 'path.to.module.file:YourModelName'
```

```{note}
You can also do this in the ``setup.cfg`` file of your repo and under ``[options.entry_points]``. Add a plugin to your
model class as follows

```{code-block} ini
[options.entry_points]
molflux.modelzoo.plugins.physicsml =
   name_of_model = path.to.module.file:YourModelName
```

This entry point allows ``molflux.modelzoo`` to hook into your model and automatically register it in the catalogue.

## The ``lightning`` module

The ``lightning`` module is where the core part of the model code lives. The general class looks like

```python
from typing import Any, Dict, Optional

import torch

from physicsml.lightning.module import PhysicsMLModuleBase
from physicsml.models.my_model.supervised.default_configs import MyModelConfig


class MyModelModule(PhysicsMLModuleBase):
    model_config: MyModelConfig

    def __init__(
        self,
        model_config: EGNNModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config=model_config)

        # configure your module code here

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # model operations on the data

        return output

    def compute_loss(self, input: Any, target: Any) -> torch.Tensor:
        # compute a loss
        return loss
```

If you model is a generic GNN, then all you have to do is to specify you module code (pure ``torch`` code), the ``forward``
pass, and the ``compute_loss`` method. The ``forward`` pass must return a dictionary which includes all the expected outputs
of the model (such as ``y_graph_scalars``, ``y_node_vector``, etc...). The ``compute_loss`` method must take in the input
``data`` batch and the output of the model and return a loss. For an example of this, see the [EGNN Module](https://github.com/Exscientia/physicsml/blob/main/src/physicsml/models/egnn/supervised/egnn_module.py).
Notice that all the lightning boiler plate is handled by the inherited ``PhysicsMLModuleBase``.

If your model requires specialised training steps (or if you'd like more control over those), then you can directly override
them as in the [ANI Module](https://github.com/Exscientia/physicsml/blob/main/src/physicsml/models/ani/supervised/ani_module.py).
