import tempfile

import molflux.modelzoo as mz
import torch


def test_training_ani(featurised_ani1x_atomic_nums):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_ani1x_atomic_nums
    # specify the model config
    model_config = {
        "name": "ani_model",  # the model name
        "config": {
            "x_features": x_features,
            "y_features": ["wb97x_dz.energy"],
            "datamodule": {
                "y_graph_scalars": ["wb97x_dz.energy"],
            },
            "which_ani": "ani2",
            "y_graph_scalars_loss_config": {
                "name": "MSELoss",
                "weight": 1.0,
            },
        },
    }

    model = mz.load_from_dict(model_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.train(
            train_data=dataset_feated,
            validation_data=dataset_feated,
            trainer_config={
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "max_epochs": 10,
                "default_root_dir": tmpdir,
            },
            datamodule_config={
                "train": {"batch_size": 4},
                "num_workers": 0,
            },
        )

    preds = model.predict(
        dataset_feated,
        trainer_config={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"},
    )

    assert "ani_model::wb97x_dz.energy" in preds.keys()
    assert len(preds) == 1
    assert len(preds["ani_model::wb97x_dz.energy"]) == 88

    assert (
        (
            torch.tensor(preds["ani_model::wb97x_dz.energy"])
            - torch.tensor(dataset_feated["wb97x_dz.energy"])
        )
        ** 2
    ).mean().sqrt() < 1.0
