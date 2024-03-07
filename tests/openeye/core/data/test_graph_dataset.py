#  type: ignore
from physicsml.lightning.graph_datasets.graph_dataset import GraphDataset


def test_graph_dataset_atom_num_only(featurised_gdb9_atomic_nums):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_gdb9_atomic_nums

    graph_dataset = GraphDataset(
        dataset=dataset_feated,
        x_features=x_features,
        y_features=None,
        with_y_features=False,
        atomic_numbers_col="physicsml_atom_numbers",
        node_attrs_col="physicsml_atom_features",
        edge_attrs_col="physicsml_bond_features",
        node_idxs_col="physicsml_atom_idxs",
        edge_idxs_col="physicsml_bond_idxs",
        coordinates_col="physicsml_coordinates",
        total_atomic_energy_col="physicsml_total_atomic_energy_col",
        num_elements=4,
        cut_off=5.0,
        y_node_scalars=None,
        y_node_vector=None,
        y_edge_scalars=None,
        y_edge_vector=None,
        y_graph_scalars=None,
        y_graph_vector=None,
        self_interaction=False,
        pbc=None,
        cell=None,
    )

    assert len(graph_dataset) == 100

    batch = graph_dataset[0]
    assert batch.coordinates.shape[0] == 5 and batch.coordinates.shape[1] == 3
    assert batch.edge_index.shape[0] == 2 and batch.edge_index.shape[1] == 20
    assert batch.num_nodes == 5
    assert batch.raw_atomic_numbers.shape[0] == 5
    assert batch.atomic_numbers.shape[0] == 5 and batch.atomic_numbers.shape[1] == 4
    assert batch["node_attrs"].shape[0] == 5 and batch["node_attrs"].shape[1] == 4

    batch = graph_dataset[1]
    assert batch.coordinates.shape[0] == 4 and batch.coordinates.shape[1] == 3
    assert batch.edge_index.shape[0] == 2 and batch.edge_index.shape[1] == 12
    assert batch.num_nodes == 4
    assert batch.raw_atomic_numbers.shape[0] == 4
    assert batch.atomic_numbers.shape[0] == 4 and batch.atomic_numbers.shape[1] == 4
    assert batch["node_attrs"].shape[0] == 4 and batch["node_attrs"].shape[1] == 4


def test_graph_dataset(featurised_gdb9_atomic_nums_and_feats_and_bond_feats):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_gdb9_atomic_nums_and_feats_and_bond_feats

    graph_dataset = GraphDataset(
        dataset=dataset_feated,
        x_features=x_features,
        y_features=None,
        with_y_features=False,
        atomic_numbers_col="physicsml_atom_numbers",
        node_attrs_col="physicsml_atom_features",
        edge_attrs_col="physicsml_bond_features",
        node_idxs_col="physicsml_atom_idxs",
        edge_idxs_col="physicsml_bond_idxs",
        coordinates_col="physicsml_coordinates",
        total_atomic_energy_col="physicsml_total_atomic_energy_col",
        num_elements=4,
        cut_off=5.0,
        y_node_scalars=None,
        y_node_vector=None,
        y_edge_scalars=None,
        y_edge_vector=None,
        y_graph_scalars=None,
        y_graph_vector=None,
        self_interaction=False,
        pbc=None,
        cell=None,
    )

    assert len(graph_dataset) == 100

    batch = graph_dataset[0]
    assert batch.coordinates.shape[0] == 5 and batch.coordinates.shape[1] == 3
    assert batch.edge_index.shape[0] == 2 and batch.edge_index.shape[1] == 20
    assert batch.num_nodes == 5
    assert batch.raw_atomic_numbers.shape[0] == 5
    assert batch.atomic_numbers.shape[0] == 5 and batch.atomic_numbers.shape[1] == 4
    assert batch["node_attrs"].shape[0] == 5 and batch["node_attrs"].shape[1] == 27
    assert batch["edge_attrs"].shape[0] == 20 and batch["edge_attrs"].shape[1] == 11

    batch = graph_dataset[1]
    assert batch.coordinates.shape[0] == 4 and batch.coordinates.shape[1] == 3
    assert batch.edge_index.shape[0] == 2 and batch.edge_index.shape[1] == 12
    assert batch.num_nodes == 4
    assert batch.raw_atomic_numbers.shape[0] == 4
    assert batch.atomic_numbers.shape[0] == 4 and batch.atomic_numbers.shape[1] == 4
    assert batch["node_attrs"].shape[0] == 4 and batch["node_attrs"].shape[1] == 27
    assert batch["edge_attrs"].shape[0] == 12 and batch["edge_attrs"].shape[1] == 11


def test_graph_dataset_no_bonds(featurised_gdb9_atomic_nums_and_feats):
    (
        dataset_feated,
        x_features,
        featurisation_metadata,
    ) = featurised_gdb9_atomic_nums_and_feats

    graph_dataset = GraphDataset(
        dataset=dataset_feated,
        x_features=x_features,
        y_features=None,
        with_y_features=False,
        atomic_numbers_col="physicsml_atom_numbers",
        node_attrs_col="physicsml_atom_features",
        edge_attrs_col="physicsml_bond_features",
        node_idxs_col="physicsml_atom_idxs",
        edge_idxs_col="physicsml_bond_idxs",
        coordinates_col="physicsml_coordinates",
        total_atomic_energy_col="physicsml_total_atomic_energy_col",
        num_elements=4,
        cut_off=5.0,
        y_node_scalars=None,
        y_node_vector=None,
        y_edge_scalars=None,
        y_edge_vector=None,
        y_graph_scalars=None,
        y_graph_vector=None,
        self_interaction=False,
        pbc=None,
        cell=None,
    )

    assert len(graph_dataset) == 100

    batch = graph_dataset[0]
    assert batch.coordinates.shape[0] == 5 and batch.coordinates.shape[1] == 3
    assert batch.edge_index.shape[0] == 2 and batch.edge_index.shape[1] == 20
    assert batch.num_nodes == 5
    assert batch.raw_atomic_numbers.shape[0] == 5
    assert batch.atomic_numbers.shape[0] == 5 and batch.atomic_numbers.shape[1] == 4
    assert batch["node_attrs"].shape[0] == 5 and batch["node_attrs"].shape[1] == 27

    batch = graph_dataset[1]
    assert batch.coordinates.shape[0] == 4 and batch.coordinates.shape[1] == 3
    assert batch.edge_index.shape[0] == 2 and batch.edge_index.shape[1] == 12
    assert batch.num_nodes == 4
    assert batch.raw_atomic_numbers.shape[0] == 4
    assert batch.atomic_numbers.shape[0] == 4 and batch.atomic_numbers.shape[1] == 4
    assert batch["node_attrs"].shape[0] == 4 and batch["node_attrs"].shape[1] == 27
