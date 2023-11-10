from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import torch
from torch.utils.data import Dataset


def validate_features(
    sub_feature: Optional[Union[List[str], str]],
    features: Optional[List[str]],
) -> Optional[Union[List[str], str]]:
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


class ANIDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        x_features: List,
        y_features: Optional[List],
        with_y_features: bool,
        y_graph_scalars: Optional[List[str]],
        y_node_vector: Optional[str],
        atomic_numbers_col: str,
        coordinates_col: str,
        total_atomic_energy_col: str,
        pbc: Optional[Tuple[bool, bool, bool]],
        cell: Optional[List[List[float]]],
    ) -> None:
        self.x_features = x_features
        self.y_features = y_features
        self.with_y_features = with_y_features

        self.atomic_numbers_col = atomic_numbers_col
        self.coordinates_col = coordinates_col
        self.total_atomic_energy_col = total_atomic_energy_col

        if pbc is not None:
            self.pbc_ten: Optional[torch.Tensor] = torch.tensor(pbc)
        else:
            self.pbc_ten = None
        if cell is not None:
            self.cell_ten: Optional[torch.Tensor] = torch.tensor(cell)
        else:
            self.cell_ten = None

        if self.with_y_features and (self.y_features is not None):
            self.y_graph_scalars = validate_features(y_graph_scalars, self.y_features)
            self.y_node_vector = validate_features(y_node_vector, self.y_features)

            dataset.set_format(columns=self.x_features + self.y_features)
        else:
            dataset.set_format(columns=self.x_features)

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def make_y_feature(
        self,
        features: Optional[Union[List[str], str]],
        datapoint: Dict,
        graph_level: bool,
    ) -> Optional[torch.Tensor]:
        if features is not None:
            if isinstance(features, list):
                y: torch.Tensor = torch.tensor(
                    [datapoint[col] for col in features],
                )
            elif isinstance(features, str):
                y = torch.tensor(datapoint[features])
            else:
                raise RuntimeError(f"Unknown feature type {type(features)}.")

            if graph_level:
                y = y.unsqueeze(0)

            return y
        else:
            return None

    def __getitem__(self, idx: int) -> Any:
        datapoint = self.dataset[idx]

        atom_numbers = datapoint[self.atomic_numbers_col]
        coordinates = datapoint[self.coordinates_col]
        total_atomic_energy = datapoint.get(self.total_atomic_energy_col, None)

        if atom_numbers is not None:
            atom_numbers = torch.tensor(atom_numbers).type(torch.int64)
        if coordinates is not None:
            coordinates = torch.tensor(coordinates)
        if total_atomic_energy is not None:
            total_atomic_energy = torch.tensor(total_atomic_energy)

        if self.with_y_features:
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
        else:
            y_graph_scalars = None
            y_node_vector = None

        return {
            "species": atom_numbers,
            "coordinates": coordinates,
            "total_atomic_energy": total_atomic_energy,
            "pbc": self.pbc_ten,
            "cell": self.cell_ten,
            "y_graph_scalars": y_graph_scalars,
            "y_node_vector": y_node_vector,
        }


def ani_collate_fn(batch_list: List) -> Dict[str, torch.Tensor]:
    batch_dict: Dict[str, Any] = {
        k: [dic[k] for dic in batch_list] for k in batch_list[0]
    }

    batch_species = torch.nn.utils.rnn.pad_sequence(batch_dict["species"], True, -1)
    batch_coords = torch.nn.utils.rnn.pad_sequence(batch_dict["coordinates"], True, 0.0)

    batch = {
        "species": batch_species,
        "coordinates": batch_coords,
    }

    if batch_dict["pbc"][0] is not None:
        batch_pbc = batch_dict["pbc"][0]
        assert all(torch.equal(x, batch_pbc) for x in batch_dict["pbc"])
        batch["pbc"] = batch_pbc
    if batch_dict["cell"][0] is not None:
        batch_cell = batch_dict["cell"][0]
        assert all(torch.equal(x, batch_cell) for x in batch_dict["cell"])
        batch["cell"] = batch_cell

    if batch_dict["y_graph_scalars"][0] is not None:
        batch_y_graph_scalars = torch.cat(batch_dict["y_graph_scalars"], dim=0)
        batch["y_graph_scalars"] = batch_y_graph_scalars

    if batch_dict["y_node_vector"][0] is not None:
        batch_y_graph_scalars = torch.cat(batch_dict["y_node_vector"], dim=0)
        batch["y_node_vector"] = batch_y_graph_scalars

    if batch_dict["total_atomic_energy"][0] is not None:
        batch_y_graph_scalars = torch.tensor(batch_dict["total_atomic_energy"])
        batch["total_atomic_energy"] = batch_y_graph_scalars

    return batch
