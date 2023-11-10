# type: ignore
from physicsml.lightning.config import PhysicsMLModelConfig
from physicsml.lightning.datamodule import PhysicsMLDataModule


def test_graph_datamodule_atom_num_only(featurised_gdb9_atomic_nums):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_gdb9_atomic_nums

    datamodule = PhysicsMLDataModule(
        model_config=PhysicsMLModelConfig(
            x_features=x_features,
            datamodule={
                "predict": {"batch_size": 4},
                "num_elements": 4,
                "cut_off": 5.0,
            },
        ),
        predict_data=dataset_feated,
    )

    loader = datamodule.predict_dataloader()
    batch = next(iter(loader))

    assert batch.coordinates.shape[0] == 16 and batch.coordinates.shape[1] == 3
    assert batch.edge_index.shape[0] == 2 and batch.edge_index.shape[1] == 50
    assert batch.num_nodes == 16
    assert batch.raw_atomic_numbers.shape[0] == 16
    assert batch.atomic_numbers.shape[0] == 16 and batch.atomic_numbers.shape[1] == 4
    assert batch["node_attrs"].shape[0] == 16 and batch["node_attrs"].shape[1] == 4
    assert batch.batch.shape[0] == 16
    assert batch.ptr.shape[0] == 5


def test_graph_datamodule(featurised_gdb9_atomic_nums_and_feats_and_bond_feats):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_gdb9_atomic_nums_and_feats_and_bond_feats

    datamodule = PhysicsMLDataModule(
        model_config=PhysicsMLModelConfig(
            x_features=x_features,
            datamodule={
                "predict": {"batch_size": 4},
                "num_elements": 4,
                "cut_off": 5.0,
            },
        ),
        predict_data=dataset_feated,
    )

    loader = datamodule.predict_dataloader()
    batch = next(iter(loader))

    assert batch.coordinates.shape[0] == 16 and batch.coordinates.shape[1] == 3
    assert batch.edge_index.shape[0] == 2 and batch.edge_index.shape[1] == 50
    assert batch.num_nodes == 16
    assert batch.raw_atomic_numbers.shape[0] == 16
    assert batch.atomic_numbers.shape[0] == 16 and batch.atomic_numbers.shape[1] == 4
    assert batch["node_attrs"].shape[0] == 16 and batch["node_attrs"].shape[1] == 27
    assert batch["edge_attrs"].shape[0] == 50 and batch["edge_attrs"].shape[1] == 12
    assert batch.batch.shape[0] == 16
    assert batch.ptr.shape[0] == 5


def test_graph_datamodule_no_bonds(featurised_gdb9_atomic_nums_and_feats):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_gdb9_atomic_nums_and_feats

    datamodule = PhysicsMLDataModule(
        model_config=PhysicsMLModelConfig(
            x_features=x_features,
            datamodule={
                "predict": {"batch_size": 4},
                "num_elements": 4,
                "cut_off": 5.0,
            },
        ),
        predict_data=dataset_feated,
    )

    loader = datamodule.predict_dataloader()
    batch = next(iter(loader))

    assert batch.coordinates.shape[0] == 16 and batch.coordinates.shape[1] == 3
    assert batch.edge_index.shape[0] == 2 and batch.edge_index.shape[1] == 50
    assert batch.num_nodes == 16
    assert batch.raw_atomic_numbers.shape[0] == 16
    assert batch.atomic_numbers.shape[0] == 16 and batch.atomic_numbers.shape[1] == 4
    assert batch["node_attrs"].shape[0] == 16 and batch["node_attrs"].shape[1] == 27
    assert batch.batch.shape[0] == 16
    assert batch.ptr.shape[0] == 5
