import tempfile

import molflux.modelzoo as mz
import torch


def test_training_nequip_energy_forces_charges(featurised_ani1x_atomic_nums):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_ani1x_atomic_nums

    # specify the model config
    model_config = {
        "name": "nequip_model",  # the model name
        "config": {
            "x_features": x_features,
            "y_features": [
                "wb97x_dz.energy",
                "wb97x_dz.hirshfeld_charges",
                "wb97x_dz.forces",
                "wb97x_dz.dipole",
            ],
            "datamodule": {
                "y_graph_scalars": ["wb97x_dz.energy"],
                "y_graph_vector": "wb97x_dz.dipole",
                "y_node_scalars": ["wb97x_dz.hirshfeld_charges"],
                "y_node_vector": "wb97x_dz.forces",
                "num_elements": 4,
            },
            "num_node_feats": 4,
            "num_layers": 2,
            "max_ell": 2,
            "parity": True,
            "num_features": 4,
            "mlp_irreps": "8x0e",
            "num_bessel": 8,
            "bessel_basis_trainable": True,
            "num_polynomial_cutoff": 6,
            "self_connection": True,
            "resnet": True,
            "avg_num_neighbours": 10.0,
            # "compute_forces": True,
            "y_graph_scalars_loss_config": {
                "name": "MSELoss",
                "weight": 1.0,
            },
            "y_graph_vector_loss_config": {
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

    assert "nequip_model::wb97x_dz.energy" in preds.keys()
    assert "nequip_model::wb97x_dz.forces" in preds.keys()
    assert "nequip_model::wb97x_dz.hirshfeld_charges" in preds.keys()
    assert "nequip_model::wb97x_dz.dipole" in preds.keys()
    assert len(preds) == 4
    assert len(preds["nequip_model::wb97x_dz.energy"]) == 88

    assert (
        (
            torch.tensor(preds["nequip_model::wb97x_dz.energy"])
            - torch.tensor(dataset_feated["wb97x_dz.energy"])
        )
        ** 2
    ).mean().sqrt() < 1.0
