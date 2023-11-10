import tempfile

import molflux.core as molflux_core
import molflux.features as prism
import molflux.modelzoo as mz
import torch
from molflux.datasets import featurise_dataset


def test_prism_allegro(featurised_gdb9_atomic_nums):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_gdb9_atomic_nums

    model_config = {
        "name": "allegro_model",  # the model name
        "config": {
            "x_features": x_features,
            "y_features": ["u0"],
            "datamodule": {
                "y_graph_scalars": ["u0"],
                "num_elements": 4,
            },
            "num_node_feats": 4,
            "num_layers": 2,
            "max_ell": 2,
            "parity": True,
            "mlp_irreps": "16x0e",
            "mlp_latent_dimensions": [12],
            "latent_mlp_latent_dimensions": [12, 12],
            "env_embed_multiplicity": 8,
            "two_body_latent_mlp_latent_dimensions": [12, 12, 12, 12],
            "num_bessel": 8,
            "bessel_basis_trainable": True,
            "num_polynomial_cutoff": 6,
            "avg_num_neighbours": 10.0,
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
                "name": "allegro",
                "config": {
                    "path": tmpdir,
                    "which_rep": "graph_embedding_mean",
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
                "name": "allegro",
                "config": {
                    "path": tmpdir,
                    "which_rep": "node_embedding",
                    "which_block": 1,
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

    assert "mol::allegro" in dataset_with_embeddings.column_names
    assert len(dataset_with_embeddings[1]["mol::allegro"]) == 12

    dataset_with_embeddings = featurise_dataset(
        dataset_feated,
        "mol",
        representation_2,
    )

    assert "mol::allegro" in dataset_with_embeddings.column_names
    assert len(dataset_with_embeddings[1]["mol::allegro"]) == 4
