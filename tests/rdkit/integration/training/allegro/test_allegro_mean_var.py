import tempfile

import molflux.modelzoo as mz
import torch


def test_training_allegro_mean_var(featurised_ani1x_atomic_nums):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_ani1x_atomic_nums

    model_config = {
        "name": "mean_var_allegro_model",  # the model name
        "config": {
            "x_features": x_features,
            "y_features": [
                "wb97x_dz.energy",
            ],
            "datamodule": {
                "y_graph_scalars": ["wb97x_dz.energy"],
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
        },
    }
    # specify the model config

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

    assert "mean_var_allegro_model::wb97x_dz.energy" in preds.keys()
    assert len(preds) == 1
    assert len(preds["mean_var_allegro_model::wb97x_dz.energy"]) == 88

    preds, stds = model.predict_with_std(  # type: ignore
        dataset_feated,
        trainer_config={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"},
    )

    assert "mean_var_allegro_model::wb97x_dz.energy" in preds.keys()
    assert "mean_var_allegro_model::wb97x_dz.energy::std" in stds.keys()
    assert len(preds) == 1
    assert len(stds) == 1
    assert len(preds["mean_var_allegro_model::wb97x_dz.energy"]) == 88
    assert len(stds["mean_var_allegro_model::wb97x_dz.energy::std"]) == 88

    preds, preds_interval = model.predict_with_prediction_interval(  # type: ignore
        dataset_feated,
        confidence=0.95,
        trainer_config={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"},
    )

    assert "mean_var_allegro_model::wb97x_dz.energy" in preds.keys()
    assert (
        "mean_var_allegro_model::wb97x_dz.energy::prediction_interval"
        in preds_interval.keys()
    )
    assert len(preds["mean_var_allegro_model::wb97x_dz.energy"]) == 88
    assert (
        len(
            preds_interval[
                "mean_var_allegro_model::wb97x_dz.energy::prediction_interval"
            ],
        )
        == 88
    )
    assert (
        len(
            preds_interval[
                "mean_var_allegro_model::wb97x_dz.energy::prediction_interval"
            ][0],
        )
        == 2
    )
