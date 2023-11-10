import tempfile

import molflux.modelzoo as mz
import torch


def test_training_tensor_net_energy_forces(featurised_ani1x_atomic_nums):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_ani1x_atomic_nums

    # specify the model config
    model_config = {
        "name": "tensor_net_model",  # the model name
        "config": {
            "x_features": x_features,
            "y_features": [
                "wb97x_dz.energy",
                "wb97x_dz.forces",
            ],
            "datamodule": {
                "y_graph_scalars": ["wb97x_dz.energy"],
                "y_node_vector": "wb97x_dz.forces",
                "num_elements": 4,
                "cut_off": 5.0,
            },
            "num_node_feats": 4,
            "num_edge_feats": 0,
            "compute_forces": True,
            "num_features": 6,
            "num_radial": 8,
            "num_interaction_layers": 3,
            "embedding_mlp_hidden_dims": [12],
            "interaction_mlp_hidden_dims": [6, 12],
            "scalar_output_mlp_hidden_dims": [6, 3],
            "y_graph_scalars_loss_config": {
                "name": "MSELoss",
                "weight": 1.0,
            },
            "y_node_vector_loss_config": {
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

    assert "tensor_net_model::wb97x_dz.energy" in preds.keys()
    assert "tensor_net_model::wb97x_dz.forces" in preds.keys()
    assert len(preds) == 2
    assert len(preds["tensor_net_model::wb97x_dz.energy"]) == 88

    assert not (
        (
            (
                torch.tensor(preds["tensor_net_model::wb97x_dz.energy"])
                - torch.tensor(dataset_feated["wb97x_dz.energy"])
            )
            ** 2
        )
        .mean()
        .sqrt()
        .isnan()
        .any()
    )
