import tempfile
from copy import deepcopy

import molflux.core as core
import molflux.modelzoo as mz
import torch

from physicsml.plugins.openmm.load import to_openmm_torchscript


def test_openmm_ani(featurised_ani1x_atomic_nums):
    dataset_feated, x_features, featurisation_metadata = featurised_ani1x_atomic_nums

    # specify the model config
    model_config = {
        "name": "ani_model",  # the model name
        "config": {
            "x_features": x_features,
            "y_features": [
                "wb97x_dz.energy",
            ],
            "datamodule": {
                "y_graph_scalars": ["wb97x_dz.energy"],
            },
            "which_ani": "ani1",
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

        core.save_model(model, tmpdir, featurisation_metadata)

        openmm_module = deepcopy(model).to_openmm(  # type: ignore
            physicsml_model=model,
            featurisation_metadata=featurisation_metadata,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        torchscript_module = to_openmm_torchscript(
            model_path=tmpdir,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        torchscript_module_32 = to_openmm_torchscript(
            model_path=tmpdir,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            precision="32",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        torchscript_module_64 = to_openmm_torchscript(
            model_path=tmpdir,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            precision="64",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    preds = model.predict(
        dataset_feated,
        trainer_config={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"},
    )

    assert "ani_model::wb97x_dz.energy" in preds.keys()
    assert len(preds) == 1
    assert len(preds["ani_model::wb97x_dz.energy"]) == 88

    pos = torch.tensor(dataset_feated[0]["physicsml_coordinates"])
    out = openmm_module(pos)
    ts_out = torchscript_module(pos)

    assert torch.allclose(
        torch.tensor(preds["ani_model::wb97x_dz.energy"][0] * 627),
        ts_out * 627,
        rtol=1e-7,
    )
    assert torch.allclose(out * 627, ts_out * 627, rtol=1e-7)

    for _ in range(20):
        assert torch.allclose(
            torch.tensor(preds["ani_model::wb97x_dz.energy"][0] * 627),
            torchscript_module_32(pos.float()) * 627,
            rtol=1e-7,
        )
        assert torch.allclose(
            torch.tensor(preds["ani_model::wb97x_dz.energy"][0] * 627).type(
                torch.float64,
            ),
            torchscript_module_64(pos.double()) * 627,
            rtol=1e-7,
        )


def test_openmm_ani_64(featurised_ani1x_atomic_nums):
    dataset_feated, x_features, featurisation_metadata = featurised_ani1x_atomic_nums

    # specify the model config
    model_config = {
        "name": "ani_model",  # the model name
        "config": {
            "x_features": x_features,
            "y_features": [
                "wb97x_dz.energy",
            ],
            "datamodule": {
                "y_graph_scalars": ["wb97x_dz.energy"],
            },
            "which_ani": "ani1",
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
                "precision": "64",
            },
            datamodule_config={
                "train": {"batch_size": 4},
                "num_workers": 0,
            },
        )

        core.save_model(model, tmpdir, featurisation_metadata)

        openmm_module = deepcopy(model).to_openmm(  # type: ignore
            physicsml_model=model,
            featurisation_metadata=featurisation_metadata,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            precision="64",
        )
        torchscript_module = to_openmm_torchscript(
            model_path=tmpdir,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="64",
        )
        torchscript_module_32 = to_openmm_torchscript(
            model_path=tmpdir,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            precision="32",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        torchscript_module_64 = to_openmm_torchscript(
            model_path=tmpdir,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            precision="64",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    preds = model.predict(
        dataset_feated,
        trainer_config={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"},
    )

    assert "ani_model::wb97x_dz.energy" in preds.keys()
    assert len(preds) == 1
    assert len(preds["ani_model::wb97x_dz.energy"]) == 88

    pos = torch.tensor(dataset_feated[0]["physicsml_coordinates"]).double()
    out = openmm_module(pos)
    ts_out = torchscript_module(pos)

    assert torch.allclose(
        torch.tensor(preds["ani_model::wb97x_dz.energy"][0] * 627).type(torch.float64),
        ts_out * 627,
        rtol=1e-7,
    )
    assert torch.allclose(out * 627, ts_out * 627, rtol=1e-7)

    for _ in range(20):
        assert torch.allclose(
            torch.tensor(preds["ani_model::wb97x_dz.energy"][0] * 627),
            torchscript_module_32(pos.float()) * 627,
            rtol=1e-7,
        )
        assert torch.allclose(
            torch.tensor(preds["ani_model::wb97x_dz.energy"][0] * 627).type(
                torch.float64,
            ),
            torchscript_module_64(pos.double()) * 627,
            rtol=1e-7,
        )


def test_openmm_ani_boxvectors(featurised_ani1x_atomic_nums):
    dataset_feated, x_features, featurisation_metadata = featurised_ani1x_atomic_nums
    # specify the model config
    model_config = {
        "name": "ani_model",  # the model name
        "config": {
            "x_features": x_features,
            "y_features": [
                "wb97x_dz.energy",
            ],
            "datamodule": {
                "y_graph_scalars": ["wb97x_dz.energy"],
                "cell": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                "pbc": (True, True, True),
            },
            "which_ani": "ani1",
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
                "max_epochs": 1,
                "default_root_dir": tmpdir,
            },
            datamodule_config={
                "train_batch_size": 4,
                "num_workers": 0,
            },
        )
        core.save_model(model, tmpdir, featurisation_metadata)
        openmm_module = deepcopy(model).to_openmm(  # type: ignore
            physicsml_model=model,
            featurisation_metadata=featurisation_metadata,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        torchscript_module = to_openmm_torchscript(
            model_path=tmpdir,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        torchscript_module_32 = to_openmm_torchscript(
            model_path=tmpdir,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            precision="32",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        torchscript_module_64 = to_openmm_torchscript(
            model_path=tmpdir,
            atom_list=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            precision="64",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    preds = model.predict(
        dataset_feated,
        trainer_config={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"},
    )
    assert "ani_model::wb97x_dz.energy" in preds.keys()
    assert len(preds) == 1
    assert len(preds["ani_model::wb97x_dz.energy"]) == 88
    pos = torch.tensor(dataset_feated[0]["physicsml_coordinates"])
    cell = torch.tensor(model.model_config.datamodule.cell)  # type: ignore
    out = openmm_module(pos, cell)
    ts_out = torchscript_module(pos, cell)
    assert torch.allclose(
        torch.tensor(preds["ani_model::wb97x_dz.energy"][0] * 627),
        ts_out * 627,
        rtol=1e-7,
    )
    assert torch.allclose(out * 627, ts_out * 627, rtol=1e-7)
    ts_out_diff_cell = torchscript_module(pos, 0.5 * cell)
    assert not torch.allclose(ts_out_diff_cell * 627, ts_out * 627, rtol=1e-7)
    for _ in range(20):
        assert torch.allclose(
            torch.tensor(preds["ani_model::wb97x_dz.energy"][0] * 627),
            torchscript_module_32(pos.float(), cell.float()) * 627,
            rtol=1e-7,
        )
        assert torch.allclose(
            torch.tensor(preds["ani_model::wb97x_dz.energy"][0] * 627).type(
                torch.float64,
            ),
            torchscript_module_64(pos.double(), cell.double()) * 627,
            rtol=1e-7,
        )
