import tempfile

import molflux.modelzoo as mz
import torch


def test_training_egnn_energy_forces_charges(featurised_ani1x_atomic_nums):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_ani1x_atomic_nums

    # specify the model config
    model_config = {
        "name": "egnn_model",  # the model name
        "config": {
            "x_features": x_features,
            "y_features": [
                "wb97x_dz.energy",
                "wb97x_dz.hirshfeld_charges",
                "wb97x_dz.forces",
            ],
            "datamodule": {
                "y_graph_scalars": ["wb97x_dz.energy"],
                "y_node_scalars": ["wb97x_dz.hirshfeld_charges"],
                "y_node_vector": "wb97x_dz.forces",
                "num_elements": 4,
                "cut_off": 5.0,
            },
            "num_node_feats": 4,
            "num_edge_feats": 0,
            "num_layers": 4,
            "num_layers_phi": 2,
            "c_hidden": 12,
            "dropout": 0.1,
            "compute_forces": True,
            "mlp_activation": "SiLU",
            "y_graph_scalars_loss_config": {
                "name": "MSELoss",
                "weight": 1.0,
            },
            "y_node_scalars_loss_config": {
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

    assert "egnn_model::wb97x_dz.energy" in preds.keys()
    assert "egnn_model::wb97x_dz.forces" in preds.keys()
    assert "egnn_model::wb97x_dz.hirshfeld_charges" in preds.keys()
    assert len(preds) == 3
    assert len(preds["egnn_model::wb97x_dz.energy"]) == 88

    assert (
        (
            torch.tensor(preds["egnn_model::wb97x_dz.energy"])
            - torch.tensor(dataset_feated["wb97x_dz.energy"])
        )
        ** 2
    ).mean().sqrt() < 1.0
