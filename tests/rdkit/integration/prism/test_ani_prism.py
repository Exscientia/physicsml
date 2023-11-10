import tempfile

import molflux.core as molflux_core
import molflux.features as prism
import molflux.modelzoo as mz
import torch
from molflux.datasets import featurise_dataset


def test_prism_ani(featurised_gdb9_atomic_nums):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_gdb9_atomic_nums

    model_config = {
        "name": "ani_model",  # the model name
        "config": {
            "x_features": x_features,
            "y_features": ["u0"],
            "datamodule": {
                "y_graph_scalars": ["u0"],
            },
            "which_ani": "ani2",
            "y_graph_scalars_loss_config": {
                "name": "MSELoss",
            },
        },
    }

    my_model = mz.load_from_dict(model_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        my_model.train(
            train_data=dataset_feated,
            validation_data=dataset_feated,
            trainer_config={
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "max_epochs": 5,
                "default_root_dir": tmpdir,
            },
            datamodule_config={
                "train": {"batch_size": 4},
                "num_workers": 0,
            },
        )
        molflux_core.save_model(my_model, tmpdir, featurisation_metadata)
        representation_1 = prism.load_from_dict(
            {
                "name": "ani",
                "config": {
                    "path": tmpdir,
                    "which_rep": "aev_sum",
                },
                "presets": {
                    "datamodule_config": {
                        "predict": {"batch_size": 12},
                    },
                    "trainer_config": {
                        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    },
                },
            },
        )
        representation_2 = prism.load_from_dict(
            {
                "name": "ani",
                "config": {
                    "path": tmpdir,
                    "which_rep": "aev",
                },
                "presets": {
                    "datamodule_config": {
                        "predict": {"batch_size": 12},
                    },
                    "trainer_config": {
                        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    },
                },
            },
        )

    dataset_with_embeddings = featurise_dataset(
        dataset_feated,
        "mol",
        representation_1,
    )

    assert "mol::ani" in dataset_with_embeddings.column_names
    assert len(dataset_with_embeddings[1]["mol::ani"]) == 1008

    dataset_with_embeddings = featurise_dataset(
        dataset_feated,
        "mol",
        representation_2,
    )

    assert "mol::ani" in dataset_with_embeddings.column_names
    assert len(dataset_with_embeddings[1]["mol::ani"]) == 8
