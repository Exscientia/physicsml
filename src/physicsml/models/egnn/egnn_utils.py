from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from physicsml.models.utils import compute_lengths_and_vectors, make_mlp


class EdgeOperation(nn.Module):
    """
    Class for edge operation
    """

    def __init__(
        self,
        num_layers_phi: int,
        c_hidden: int,
        edge_feats_bool: bool,
        dropout: Optional[float],
        mlp_activation: Optional[str],
        mlp_output_activation: Optional[str],
        num_rbf: int,
        bessel_cut_off: float,
    ) -> None:
        super().__init__()

        self.with_rbf = num_rbf > 0
        if self.with_rbf:
            self.bessel_cut_off: Optional[torch.nn.Parameter] = torch.nn.Parameter(
                torch.Tensor([bessel_cut_off]),
            )
            self.z_0k: Optional[torch.nn.Parameter] = torch.nn.Parameter(
                torch.pi * torch.arange(num_rbf),
            )
        else:
            self.bessel_cut_off = None
            self.z_0k = None

        # MLP for phi_e
        if edge_feats_bool:
            c_in = 3 * c_hidden + num_rbf + 1
        else:
            c_in = 2 * c_hidden + num_rbf + 1

        self.phi_e = make_mlp(
            c_in=c_in,
            c_hidden=c_hidden,
            c_out=c_hidden,
            num_layers=num_layers_phi,
            dropout=dropout,
            mlp_activation=mlp_activation,
            output_activation=mlp_output_activation,
        )

        # MLP for phi_b
        if edge_feats_bool:
            self.phi_b = make_mlp(
                c_in=4 * c_hidden + num_rbf + 1,
                c_hidden=c_hidden,
                c_out=c_hidden,
                num_layers=num_layers_phi,
                dropout=dropout,
                mlp_activation=mlp_activation,
                output_activation=mlp_output_activation,
            )
        else:
            self.phi_b = None  # type: ignore

    def forward(
        self,
        node_feats: torch.Tensor,
        coordinates: torch.Tensor,
        edge_feats: Optional[torch.Tensor],
        edge_indices: torch.Tensor,
        cell: Optional[torch.Tensor],
        cell_shift_vector: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sender = edge_indices[0]
        receiver = edge_indices[1]

        node_feats_i = node_feats[sender]
        node_feats_j = node_feats[receiver]

        abs_r_ji, _ = compute_lengths_and_vectors(
            positions=coordinates,
            edge_index=edge_indices,
            cell=cell,
            cell_shift_vector=cell_shift_vector,
        )

        tens_to_cat = [node_feats_i, node_feats_j, abs_r_ji]

        if (
            self.with_rbf
            and (self.bessel_cut_off is not None)
            and (self.z_0k is not None)
        ):
            # find bessel functions
            rbf_ji = (
                torch.sqrt(2.0 / self.bessel_cut_off)
                * torch.sin(self.z_0k * abs_r_ji / self.bessel_cut_off)
                / abs_r_ji
            )
            tens_to_cat.append(rbf_ji)

        if edge_feats is not None:
            tens_to_cat.append(edge_feats)

        # concatenate and find messages
        m_ji: torch.Tensor = self.phi_e(torch.cat(tens_to_cat, dim=-1))

        # update edge feats
        if self.phi_b is not None:
            tens_to_cat = [node_feats_i, node_feats_j, m_ji, abs_r_ji]

            if (
                self.with_rbf
                and (self.bessel_cut_off is not None)
                and (self.z_0k is not None)
            ):
                tens_to_cat.append(rbf_ji)

            if edge_feats is not None:
                tens_to_cat.append(edge_feats)
            edge_feats = self.phi_b(torch.cat(tens_to_cat, dim=-1))

        return m_ji, edge_feats


class NodeOperation(nn.Module):
    """
    Class for node operation
    """

    def __init__(
        self,
        num_layers_phi: int,
        c_hidden: int,
        dropout: Optional[float],
        mlp_activation: Optional[str],
        mlp_output_activation: Optional[str],
        modify_coords: int,
    ) -> None:
        super().__init__()

        self.modify_coords = modify_coords

        # MLP for phi_n
        self.phi_n = make_mlp(
            c_in=2 * c_hidden,
            c_hidden=c_hidden,
            c_out=c_hidden,
            num_layers=num_layers_phi,
            dropout=dropout,
            mlp_activation=mlp_activation,
            output_activation=mlp_output_activation,
        )

        # MLP for attention
        self.attention = make_mlp(
            c_in=c_hidden,
            c_hidden=c_hidden,
            c_out=1,
            num_layers=num_layers_phi,
            dropout=dropout,
            mlp_activation=mlp_activation,
            output_activation="Sigmoid",
        )

        # MLP for phi_x
        if self.modify_coords:
            self.phi_x: Optional[torch.nn.Sequential] = make_mlp(
                c_in=c_hidden,
                c_hidden=c_hidden,
                c_out=1,
                num_layers=num_layers_phi,
                mlp_activation=mlp_activation,
                output_activation=None,
            )
        else:
            self.phi_x = None

    def forward(
        self,
        node_feats: torch.Tensor,
        coordinates: torch.Tensor,
        m_ji: torch.Tensor,
        edge_indices: torch.Tensor,
        cell: Optional[torch.Tensor],
        cell_shift_vector: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # number of all nodes in batch
        num_nodes = node_feats.shape[0]
        receiver = edge_indices[0]

        # find attentions
        attention_ji = self.attention(m_ji)

        # aggregate messages * attentions
        m_i = scatter(
            src=m_ji * attention_ji,
            index=receiver,
            dim=0,
            dim_size=num_nodes,
        )

        # update nodes with residual connection
        node_feats = self.phi_n(torch.cat([node_feats, m_i], dim=-1)) + node_feats
        if self.modify_coords and (self.phi_x is not None):
            abs_r_ji, r_ji = compute_lengths_and_vectors(
                positions=coordinates,
                edge_index=edge_indices,
                cell=cell,
                cell_shift_vector=cell_shift_vector,
            )

            delta_coords_ji = r_ji / (abs_r_ji + 1) * self.phi_x(m_ji)
            delta_coords_i = scatter(
                src=delta_coords_ji,
                index=receiver,
                dim=0,
                dim_size=num_nodes,
            )
            coordinates = coordinates + delta_coords_i

        return node_feats, coordinates, attention_ji


class EGNNBlock(nn.Module):
    """
    Class for block of egnn
    """

    def __init__(
        self,
        num_layers_phi: int,
        c_hidden: int,
        edge_feats_bool: bool,
        dropout: Optional[float],
        mlp_activation: Optional[str],
        mlp_output_activation: Optional[str],
        modify_coords: bool,
        num_rbf: int,
        bessel_cut_off: float,
    ) -> None:
        super().__init__()

        self.modify_coords = modify_coords
        self.edge_feats_bool = edge_feats_bool

        self.edge_op = EdgeOperation(
            num_layers_phi=num_layers_phi,
            c_hidden=c_hidden,
            edge_feats_bool=edge_feats_bool,
            dropout=dropout,
            mlp_activation=mlp_activation,
            mlp_output_activation=mlp_output_activation,
            num_rbf=num_rbf,
            bessel_cut_off=bessel_cut_off,
        )
        self.node_op = NodeOperation(
            num_layers_phi=num_layers_phi,
            c_hidden=c_hidden,
            dropout=dropout,
            mlp_activation=mlp_activation,
            mlp_output_activation=mlp_output_activation,
            modify_coords=self.modify_coords,
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """

        Args:
            data: dict

        Returns: dict

        """

        node_feats = data["node_feats"]
        coordinates = data["coordinates"]
        if self.edge_feats_bool:
            edge_feats = data["edge_feats"]
        else:
            edge_feats = None

        if "cell" in data:
            cell = data["cell"]
            cell_shift_vector = data["cell_shift_vector"]
        else:
            cell = None
            cell_shift_vector = None

        edge_indices = data["edge_index"]

        m_ji, edge_feats = self.edge_op(
            node_feats=node_feats,
            coordinates=coordinates,
            edge_feats=edge_feats,
            edge_indices=edge_indices,
            cell=cell,
            cell_shift_vector=cell_shift_vector,
        )

        node_feats, coordinates, attention_ji = self.node_op(
            node_feats=node_feats,
            coordinates=coordinates,
            m_ji=m_ji,
            edge_indices=edge_indices,
            cell=cell,
            cell_shift_vector=cell_shift_vector,
        )

        data["node_feats"] = node_feats
        data["coordinates"] = coordinates
        if edge_feats is not None:
            data["edge_feats"] = edge_feats
        data["attention"] = attention_ji

        return data


class EGNN(torch.nn.Module):
    """
    Class for egnn model
    """

    def __init__(
        self,
        num_node_feats: int,
        num_edge_feats: int,
        num_layers: int,
        num_layers_phi: int,
        c_hidden: int,
        dropout: Optional[float],
        mlp_activation: Optional[str],
        mlp_output_activation: Optional[str],
        num_rbf: int,
        modify_coords: bool,
        bessel_cut_off: float,
    ) -> None:
        super().__init__()

        # node embedding
        self.embed_nodes = make_mlp(
            c_in=num_node_feats,
            c_hidden=c_hidden,
            c_out=c_hidden,
            num_layers=num_layers_phi,
            dropout=dropout,
            mlp_activation=mlp_activation,
            output_activation=mlp_output_activation,
        )

        # edge embedding (if there are edge feats)
        edge_feats_bool = num_edge_feats > 0
        if edge_feats_bool:
            self.embed_edges = make_mlp(
                c_in=num_edge_feats,
                c_hidden=c_hidden,
                c_out=c_hidden,
                num_layers=num_layers_phi,
                dropout=dropout,
                mlp_activation=mlp_activation,
                output_activation=mlp_output_activation,
            )
        else:
            self.embed_edges = None  # type: ignore

        self.egnn_embedding_block = EGNNBlock(
            num_layers_phi=num_layers_phi,
            c_hidden=c_hidden,
            edge_feats_bool=edge_feats_bool,
            dropout=dropout,
            mlp_activation=mlp_activation,
            mlp_output_activation=mlp_output_activation,
            modify_coords=modify_coords,
            num_rbf=num_rbf,
            bessel_cut_off=bessel_cut_off,
        )

        self.egnn_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.egnn_blocks.append(
                EGNNBlock(
                    num_layers_phi=num_layers_phi,
                    c_hidden=c_hidden,
                    dropout=dropout,
                    mlp_activation=mlp_activation,
                    mlp_output_activation=mlp_output_activation,
                    modify_coords=modify_coords,
                    num_rbf=num_rbf,
                    bessel_cut_off=bessel_cut_off,
                    edge_feats_bool=edge_feats_bool,
                ),
            )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """

        Args:
            data: dict

        Returns: dict

        """

        data["node_feats"] = self.embed_nodes(data["node_attrs"])
        if self.embed_edges is not None:
            data["edge_feats"] = self.embed_edges(data["edge_attrs"])

        data = self.egnn_embedding_block(data)

        for _layer_idx, layer in enumerate(self.egnn_blocks):
            data = layer(data)

        return data


class PoolingHead(torch.nn.Module):
    """
    Class for EGNN model pooling head
    """

    def __init__(
        self,
        c_hidden: int,
        num_layers_phi: int,
        pool_type: Literal["sum", "mean"],
        pool_from: Literal["nodes", "edges", "nodes_edges"],
        num_tasks: Optional[int],
        dropout: Optional[float],
        mlp_activation: Optional[str],
        mlp_output_activation: Optional[str],
        output_activation: Optional[str],
    ) -> None:
        super().__init__()

        assert num_tasks is not None

        # MLPs for prediction
        self.headMLP1 = make_mlp(
            c_in=c_hidden,
            c_hidden=c_hidden,
            c_out=c_hidden,
            num_layers=num_layers_phi,
            dropout=dropout,
            mlp_activation=mlp_activation,
            output_activation=mlp_output_activation,
        )

        list_of_pools = ["sum", "mean"]
        assert pool_type in list_of_pools, KeyError(
            f"{pool_type} not available. Choose from {list_of_pools}",
        )
        self.pool_type = pool_type

        self.pool_from = pool_from
        if self.pool_from == "nodes_edges":
            c_in_head_2 = 2 * c_hidden
        else:
            c_in_head_2 = c_hidden

        self.headMLP2 = make_mlp(
            c_in=c_in_head_2,
            c_hidden=c_hidden,
            c_out=num_tasks,
            num_layers=num_layers_phi,
            dropout=dropout,
            mlp_activation=mlp_activation,
            output_activation=output_activation,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """

        Args:
            data: dict of features.

        Returns: pooled output (bs, 1)

        """

        node_feats = data["node_feats"]
        num_nodes = node_feats.shape[0]

        list_of_pooled_feats: List[torch.Tensor] = []
        if self.pool_from in ["nodes", "nodes_edges"]:
            node_feats = self.headMLP1(node_feats)
            pooled_node_feats = scatter(
                src=node_feats,
                index=data["batch"],
                dim=0,
                dim_size=int(data["num_graphs"]),
                reduce=self.pool_type,
            )

            list_of_pooled_feats.append(pooled_node_feats)

        if self.pool_from in ["edges", "nodes_edges"]:
            edge_feats = data["edge_feats"]
            edge_feats = self.headMLP1(edge_feats)

            receiver = data["edge_index"][1]

            pooled_edge_feats = scatter(
                src=edge_feats,
                index=receiver,
                dim=0,
                dim_size=num_nodes,
                reduce=self.pool_type,
            )
            pooled_edge_feats = scatter(
                src=pooled_edge_feats,
                index=data["batch"],
                dim=0,
                dim_size=int(data["num_graphs"]),
                reduce=self.pool_type,
            )  # [n_graphs,]

            list_of_pooled_feats.append(pooled_edge_feats)

        pooled_feats: torch.Tensor = self.headMLP2(
            torch.cat(
                list_of_pooled_feats,
                dim=-1,
            ),  # [batch_size, dim_feats] -> [batch_size, num_tasks]
        )

        return pooled_feats
