import datasets
from torch_geometric.data import Dataset

from physicsml.lightning.graph_datasets.graph_dataset import (
    GraphDataset,
    GraphDatum,
)


class MultiGraphDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        x_features: list,
        y_features: list | None,
        train_features: list | None,
        with_y_features: bool,
        graph_names: list[str],
        dict_atomic_numbers_col: dict[str, str],
        dict_node_attrs_col: dict[str, str],
        dict_edge_attrs_col: dict[str, str],
        dict_node_idxs_col: dict[str, str],
        dict_edge_idxs_col: dict[str, str],
        dict_coordinates_col: dict[str, str],
        y_graph_scalars: list[str] | None,
        num_elements: int,
        self_interaction: bool,
        pbc: tuple[bool, bool, bool] | None,
        cell: list | None,
        cut_off: float,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.graph_datasets = {}

        for graph_name in graph_names:
            self.graph_datasets[graph_name] = GraphDataset(
                dataset=dataset,
                x_features=x_features,
                y_features=y_features,
                with_y_features=with_y_features,
                atomic_numbers_col=dict_atomic_numbers_col[graph_name],
                node_attrs_col=dict_node_attrs_col[graph_name],
                edge_attrs_col=dict_edge_attrs_col[graph_name],
                node_idxs_col=dict_node_idxs_col[graph_name],
                edge_idxs_col=dict_edge_idxs_col[graph_name],
                graph_attrs_cols=None,
                coordinates_col=dict_coordinates_col[graph_name],
                total_atomic_energy_col=None,  # type: ignore
                y_node_scalars=None,
                y_node_vector=None,
                y_edge_scalars=None,
                y_edge_vector=None,
                y_graph_scalars=y_graph_scalars,
                y_graph_vector=None,
                num_elements=num_elements,
                self_interaction=self_interaction,
                pbc=pbc,
                cell=cell,
                cut_off=cut_off,
            )

    def len(self) -> int:
        return len(self.dataset)

    def get(self, idx: int) -> dict[str, GraphDatum]:
        output = {}
        for k, v in self.graph_datasets.items():
            output[k] = v[idx]

        return output

    def __repr__(self) -> str:
        return f"MultiGraphDataset({[(k, len(v)) for k, v in self.graph_datasets.items()]})"
