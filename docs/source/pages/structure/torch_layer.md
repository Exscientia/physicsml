# Torch layer

The lowest layer is ``torch``. Here, there are only two configs we need to worry about: the ``optimizer`` and ``scheduler``
configs.


## ``optimizer`` config

This is config for the optimizer. It can be used to construct any optimizer available under ``torch.optim``. The config
has the following structure

* ``name``: The name of optimizer (from the ``torch.optim`` module).
* ``config``: A dict of the kwargs the optimizer takes.


## ``scheduler`` config

This is config for the scheduler. It can be used to construct any scheduler available under ``torch.optim.lr_scheduler``.
The config has the following structure

* ``name``: The name of scheduler (from the ``torch.optim.lr_scheduler`` module).
* ``config``: A dict of the kwargs the scheduler takes.
* ``interval``: The interval to use (for schedulers like ``ReduceLROnPlateua``).
* ``frequency``: The interval frequency to use (for schedulers like ``ReduceLROnPlateua``).
* ``monitor``: The metric to monitor (for schedulers like ``ReduceLROnPlateua``).
* ``strict``: Whether to fail or raise a warning if monitoring metric not found (for schedulers like ``ReduceLROnPlateua``).
