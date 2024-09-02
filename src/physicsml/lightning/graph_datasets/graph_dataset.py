import datasets
import torch
import torch.utils.data
from torch_geometric.data import Data, Dataset

from physicsml.lightning.graph_datasets.neighbourhood_list_torch import (
    construct_edge_indices_and_attrs,
)


class GraphDatum(Data):
    num_graphs: torch.Tensor | None
    batch: torch.Tensor | None
    raw_atomic_numbers: torch.Tensor | None
    atomic_numbers: torch.Tensor | None
    edge_index: torch.Tensor | None
    node_attrs: torch.Tensor | None
    edge_attrs: torch.Tensor | None
    graph_attrs: torch.Tensor | None
    y_node_scalars: torch.Tensor | None
    y_node_vector: torch.Tensor | None
    y_edge_scalars: torch.Tensor | None
    y_edge_vector: torch.Tensor | None
    y_graph_scalars: torch.Tensor | None
    y_graph_vector: torch.Tensor | None
    cell: torch.Tensor | None
    total_atomic_energy: torch.Tensor | None
    coordinates: torch.Tensor | None
    cell_shift_vector: torch.Tensor | None

    def __init__(
        self,
        raw_atomic_numbers: torch.Tensor | None,  # [n_nodes]
        atomic_numbers: torch.Tensor | None,  # [n_nodes]
        edge_index: torch.Tensor | None,  # [2, n_edges]
        node_attrs: torch.Tensor | None,  # [n_nodes, n_node_attrs]
        edge_attrs: torch.Tensor | None,  # [n_edges, n_edge_attrs]
        graph_attrs: torch.Tensor | None,  # [1, n_graph_attrs]
        y_node_scalars: torch.Tensor | None,  # [n_nodes, n_scalars]
        y_node_vector: torch.Tensor | None,  # [n_nodes, dim_vector]
        y_edge_scalars: torch.Tensor | None,  # [n_edges, n_scalars]
        y_edge_vector: torch.Tensor | None,  # [n_edges, dim_vector]
        y_graph_scalars: torch.Tensor | None,  # [n_scalars]
        y_graph_vector: torch.Tensor | None,  # [dim_vector]
        cell: torch.Tensor | None,  # [3, 3]
        total_atomic_energy: torch.Tensor | None,  # [1,]
        coordinates: torch.Tensor | None,  # [n_nodes, 3]
        cell_shift_vector: torch.Tensor | None,  # [n_nodes, 3]
    ) -> None:
        if raw_atomic_numbers is not None:
            num_nodes = raw_atomic_numbers.shape[0]
        else:
            num_nodes = None

        # Check shapes
        assert (coordinates is None) or (coordinates.shape[1] == 3)
        assert (cell_shift_vector is None) or (cell_shift_vector.shape[1] == 3)

        assert y_node_scalars is None or (
            (len(y_node_scalars.shape) == 2) and y_node_scalars.shape[0] == num_nodes
        )
        assert y_edge_scalars is None or (
            (len(y_edge_scalars.shape) == 2)
            and y_edge_scalars.shape[0] == edge_index.shape[1]  # type: ignore
        )
        assert y_graph_scalars is None or (len(y_graph_scalars.shape) == 2)

        assert y_node_vector is None or (
            (len(y_node_vector.shape) == 2) and y_node_vector.shape[0] == num_nodes
        )
        assert y_edge_vector is None or (
            (len(y_edge_vector.shape) == 2)
            and y_edge_vector.shape[0] == edge_index.shape[1]  # type: ignore
        )
        assert y_graph_vector is None or (len(y_graph_vector.shape) == 2)

        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "raw_atomic_numbers": raw_atomic_numbers,
            "atomic_numbers": atomic_numbers,
            "edge_index": edge_index,
            "node_attrs": node_attrs,
            "edge_attrs": edge_attrs,
            "graph_attrs": graph_attrs,
            "y_node_scalars": y_node_scalars,
            "y_node_vector": y_node_vector,
            "y_edge_scalars": y_edge_scalars,
            "y_edge_vector": y_edge_vector,
            "y_graph_scalars": y_graph_scalars,
            "y_graph_vector": y_graph_vector,
            "cell": cell,
            "total_atomic_energy": total_atomic_energy,
            "coordinates": coordinates,
            "cell_shift_vector": cell_shift_vector,
        }
        super().__init__(**data)

    def is_node_attr(self, key: str) -> bool:
        if key in [
            "raw_atomic_numbers",
            "atomic_numbers",
            "node_attrs",
            "y_node_scalars",
            "y_node_vector",
            "coordinates",
        ]:
            return True
        else:
            return False

    def is_edge_attr(self, key: str) -> bool:
        if key in [
            "edge_attrs",
            "y_edge_scalars",
            "y_edge_vector",
            "cell_shift_vector",
        ]:
            return True
        else:
            return False


def validate_features(
    sub_feature: list[str] | str | None,
    features: list[str] | None,
) -> list[str] | str | None:
    # assert that all cols exist in y_features and sort features
    if (sub_feature is not None) and (features is not None):
        if isinstance(sub_feature, list):
            assert len(sub_feature) > 0
            assert all(col in features for col in sub_feature)
            sub_feature = [col for col in features if col in sub_feature]
        elif isinstance(sub_feature, str):
            assert sub_feature in features
        else:
            raise KeyError(
                f"Unknown feature {sub_feature} with type {type(sub_feature)}.",
            )

    return sub_feature


class GraphDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        x_features: list,
        y_features: list | None,
        with_y_features: bool,
        atomic_numbers_col: str,
        node_attrs_col: str,
        edge_attrs_col: str,
        node_idxs_col: str,
        edge_idxs_col: str,
        graph_attrs_cols: list[str] | None,
        coordinates_col: str,
        total_atomic_energy_col: str,
        y_node_scalars: list[str] | None,
        y_node_vector: str | None,
        y_edge_scalars: list[str] | None,
        y_edge_vector: str | None,
        y_graph_scalars: list[str] | None,
        y_graph_vector: str | None,
        num_elements: int,
        self_interaction: bool,
        pbc: tuple[bool, bool, bool] | None,
        cell: list | None,
        cut_off: float,
    ) -> None:
        super().__init__()

        self.x_features = x_features
        self.y_features = y_features
        self.with_y_features = with_y_features

        # columns
        self.atomic_numbers_col = atomic_numbers_col
        self.node_attrs_col = node_attrs_col
        self.edge_attrs_col = edge_attrs_col
        self.node_idxs_col = node_idxs_col
        self.edge_idxs_col = edge_idxs_col
        self.graph_attrs_cols = graph_attrs_cols
        # sort alphabetically
        if self.graph_attrs_cols is not None:
            self.graph_attrs_cols = sorted(self.graph_attrs_cols)

        self.coordinates_col = coordinates_col
        self.total_atomic_energy_col = total_atomic_energy_col

        # num elements
        self.num_elements = num_elements

        # neighbourhood setup
        self.cutoff = cut_off
        self.pbc = pbc
        if cell is not None:
            self.cell_ten: torch.Tensor | None = torch.tensor(cell)
        else:
            self.cell_ten = None
        self.self_interaction = self_interaction

        graph_dataset_features = self.x_features
        if self.with_y_features and (self.y_features is not None):
            # assert that all cols exist in y_features and sort features
            self.y_node_scalars = validate_features(y_node_scalars, self.y_features)
            self.y_edge_scalars = validate_features(y_edge_scalars, self.y_features)
            self.y_graph_scalars = validate_features(y_graph_scalars, self.y_features)

            self.y_node_vector = validate_features(y_node_vector, self.y_features)
            self.y_edge_vector = validate_features(y_edge_vector, self.y_features)
            self.y_graph_vector = validate_features(y_graph_vector, self.y_features)

            graph_dataset_features += self.y_features

        dataset.set_format(columns=graph_dataset_features)
        self.dataset = dataset

    def len(self) -> int:
        return len(self.dataset)

    def make_y_feature(
        self,
        features: list[str] | str | None,
        datapoint: dict,
        graph_level: bool,
    ) -> torch.Tensor | None:
        if features is not None:
            if isinstance(features, list):
                y: torch.Tensor = torch.tensor(
                    [datapoint[col] for col in features],
                )
                if not graph_level:
                    y = y.transpose(0, 1)
            elif isinstance(features, str):
                y = torch.tensor(datapoint[features])
            else:
                raise RuntimeError(f"Unknown feature type {type(features)}.")

            if graph_level:
                y = y.unsqueeze(0)

            return y
        else:
            return None

    def get(self, idx: int) -> GraphDatum:
        datapoint = self.dataset[idx]

        # Extract data from datapoint
        raw_atomic_numbers = datapoint.get(self.atomic_numbers_col, None)
        node_attrs = datapoint.get(self.node_attrs_col, None)
        initial_edge_attrs = datapoint.get(self.edge_attrs_col, None)
        initial_edge_indices = datapoint.get(self.edge_idxs_col, None)
        coordinates = datapoint[self.coordinates_col]
        total_atomic_energy = datapoint.get(self.total_atomic_energy_col, None)

        if self.graph_attrs_cols is not None:
            graph_attrs_list = []
            for graph_attrs_col in self.graph_attrs_cols:
                feat = datapoint[graph_attrs_col]
                if isinstance(feat, list):
                    graph_attrs_list += feat
                elif isinstance(feat, float) or isinstance(feat, int):
                    graph_attrs_list.append(feat)
                else:
                    raise RuntimeError(
                        f"Unsupported type {type(feat)} for graph attributes.",
                    )
            graph_attrs = torch.tensor(graph_attrs_list).unsqueeze(0) * 1.0
        else:
            graph_attrs = None

        if raw_atomic_numbers is not None:
            raw_atomic_numbers = torch.tensor(raw_atomic_numbers)
            atomic_numbers: torch.Tensor | None = (
                torch.nn.functional.one_hot(
                    raw_atomic_numbers,
                    num_classes=self.num_elements,
                )
                * 1.0
            )
        else:
            atomic_numbers = None

        if total_atomic_energy is not None:
            total_atomic_energy = torch.tensor(total_atomic_energy).unsqueeze(0)

        if coordinates is not None:
            coordinates = torch.tensor(coordinates)

        if (node_attrs is not None) and (atomic_numbers is not None):
            node_attrs = (
                torch.cat([atomic_numbers, torch.tensor(node_attrs)], dim=1) * 1.0
            )
        elif atomic_numbers is not None:
            node_attrs = atomic_numbers * 1.0
        else:
            node_attrs = None

        if initial_edge_attrs == []:
            # dataset will return list when in torch format with empty edge_attrs
            initial_edge_attrs = torch.empty(0).float()
        elif initial_edge_attrs is not None:
            initial_edge_attrs = torch.tensor(initial_edge_attrs).float()

        if initial_edge_indices == []:
            # if no edge indices, add empty tensor in the same shape
            initial_edge_indices = torch.empty(0, 2).type(torch.int64)
        elif initial_edge_indices is not None:
            initial_edge_indices = torch.tensor(initial_edge_indices).type(torch.int64)

        # Construct edge indices
        edge_indices, edge_attrs, cell_shift_vector = construct_edge_indices_and_attrs(
            positions=coordinates,
            initial_edge_indices=initial_edge_indices,
            initial_edge_attrs=initial_edge_attrs,
            pbc=self.pbc,
            cell=self.cell_ten,
            cutoff=self.cutoff,
            self_interaction=self.self_interaction,
        )

        if edge_attrs is not None:
            edge_attrs = edge_attrs * 1.0

        if edge_indices is not None:
            edge_indices = edge_indices.type(torch.int64)

        if self.with_y_features:
            y_node_scalars = self.make_y_feature(
                self.y_node_scalars,
                datapoint,
                graph_level=False,
            )
            y_edge_scalars = self.make_y_feature(
                self.y_edge_scalars,
                datapoint,
                graph_level=False,
            )
            y_graph_scalars = self.make_y_feature(
                self.y_graph_scalars,
                datapoint,
                graph_level=True,
            )
            y_node_vector = self.make_y_feature(
                self.y_node_vector,
                datapoint,
                graph_level=False,
            )
            y_edge_vector = self.make_y_feature(
                self.y_edge_vector,
                datapoint,
                graph_level=False,
            )
            y_graph_vector = self.make_y_feature(
                self.y_graph_vector,
                datapoint,
                graph_level=True,
            )
        else:
            y_node_scalars = None
            y_edge_scalars = None
            y_graph_scalars = None
            y_node_vector = None
            y_edge_vector = None
            y_graph_vector = None

        return GraphDatum(
            raw_atomic_numbers=raw_atomic_numbers,
            atomic_numbers=atomic_numbers,
            total_atomic_energy=total_atomic_energy,
            coordinates=coordinates,
            edge_index=edge_indices,
            cell_shift_vector=cell_shift_vector,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            graph_attrs=graph_attrs,
            y_node_scalars=y_node_scalars,
            y_node_vector=y_node_vector,
            y_edge_scalars=y_edge_scalars,
            y_edge_vector=y_edge_vector,
            y_graph_scalars=y_graph_scalars,
            y_graph_vector=y_graph_vector,
            cell=self.cell_ten,
        )
