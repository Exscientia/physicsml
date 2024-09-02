from typing import Literal

import torch
from torch_geometric.utils import scatter

from physicsml.models.utils import make_mlp


class LigandPocketDiffPoolingHead(torch.nn.Module):
    """
    Class for MultiGraphPoolingHead
    """

    def __init__(
        self,
        c_hidden: int,
        num_layers_phi: int,
        pool_type: Literal["sum", "mean"],
        pool_from: Literal["nodes", "edges", "nodes_edges"],
        num_tasks: int | None,
        dropout: float | None,
        mlp_activation: str | None,
        mlp_output_activation: str | None,
        output_activation: str | None,
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
        data: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """

        Args:
            data: dict of features.

        Returns: pooled output (bs, 1)

        """
        pooled_outs = {}
        for k, v in data.items():
            node_feats = v["node_feats"]
            num_nodes = node_feats.shape[0]

            list_of_pooled_feats: list[torch.Tensor] = []
            if self.pool_from in ["nodes", "nodes_edges"]:
                node_feats = self.headMLP1(node_feats)
                pooled_node_feats = scatter(
                    src=node_feats,
                    index=v["batch"],
                    dim=0,
                    dim_size=int(v["num_graphs"]),
                    reduce=self.pool_type,
                )

                list_of_pooled_feats.append(pooled_node_feats)

            if self.pool_from in ["edges", "nodes_edges"]:
                edge_feats = v["edge_feats"]
                edge_feats = self.headMLP1(edge_feats)

                receiver = v["edge_index"][1]

                pooled_edge_feats = scatter(
                    src=edge_feats,
                    index=receiver,
                    dim=0,
                    dim_size=num_nodes,
                    reduce=self.pool_type,
                )
                pooled_edge_feats = scatter(
                    src=pooled_edge_feats,
                    index=v["batch"],
                    dim=0,
                    dim_size=int(v["num_graphs"]),
                    reduce=self.pool_type,
                )  # [n_graphs,]

                list_of_pooled_feats.append(pooled_edge_feats)

            pooled_feats: torch.Tensor = self.headMLP2(
                torch.cat(
                    list_of_pooled_feats,
                    dim=-1,
                ),
            )
            pooled_outs[k] = pooled_feats

        output: torch.Tensor = (
            pooled_outs["ligand_pocket"] - pooled_outs["pocket"] - pooled_outs["ligand"]
        )
        return output


class LigandPocketPoolingHead(torch.nn.Module):
    """
    Class for LigandPocketPoolingHead
    """

    def __init__(
        self,
        c_hidden: int,
        num_layers_phi: int,
        pool_type: Literal["sum", "mean"],
        pool_from: Literal["nodes", "edges", "nodes_edges"],
        num_tasks: int | None,
        dropout: float | None,
        mlp_activation: str | None,
        mlp_output_activation: str | None,
        output_activation: str | None,
    ) -> None:
        super().__init__()

        assert num_tasks is not None

        list_of_pools = ["sum", "mean"]
        assert pool_type in list_of_pools, KeyError(
            f"{pool_type} not available. Choose from {list_of_pools}",
        )
        self.pool_type = pool_type

        self.interaction_coefs_mlp = make_mlp(
            c_in=c_hidden,
            c_hidden=c_hidden,
            c_out=1,
            num_layers=num_layers_phi,
            dropout=dropout,
            mlp_activation=mlp_activation,
            output_activation=output_activation,
        )

        self.lin_down = torch.nn.Linear(
            in_features=c_hidden,
            out_features=c_hidden // 6,
            bias=False,
        )

        self.headMLP2 = make_mlp(
            c_in=2 * (12 * (c_hidden // 6) + 9),
            c_hidden=c_hidden,
            c_out=num_tasks,
            num_layers=num_layers_phi,
            dropout=dropout,
            mlp_activation=mlp_activation,
            output_activation=output_activation,
        )

    def get_eigen_vectors(
        self,
        vectors: torch.Tensor,
        graph_data: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # get vectors com as mean of vectors
        com = scatter(
            src=vectors,
            index=graph_data["batch"],
            dim=0,
            dim_size=int(graph_data["num_graphs"]),
            reduce="mean",
        )

        # get com vectors
        com_vectors = vectors - com[graph_data["batch"]]
        normed_com_vectors = com_vectors / com_vectors.norm(dim=-1, keepdim=True)

        # get XtX matrix for PCA
        XTX = scatter(
            src=com_vectors.unsqueeze(1) * com_vectors.unsqueeze(2),
            index=graph_data["batch"],
            dim=0,
            dim_size=int(graph_data["num_graphs"]),
            reduce="mean",
        )

        # get eigenvectors
        decomp = torch.linalg.eig(XTX)

        # sort by magnitude of eigen vals
        sorted_idxs = decomp.eigenvalues.real.argsort(1)
        slice_idxs = torch.arange(sorted_idxs.shape[0], device=sorted_idxs.device)

        eig_vecs = decomp.eigenvectors.real.transpose(1, 2)
        eig_vecs = eig_vecs[:, sorted_idxs][slice_idxs, slice_idxs]

        return {
            "eig_vecs": eig_vecs,
            "com_vectors": com_vectors,
            "normed_com_vectors": normed_com_vectors,
        }

    def get_projected_feats(
        self,
        feats: torch.Tensor,
        com_normed_vectors: torch.Tensor,
        eig_vecs: torch.Tensor,
        graph_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        coefs = (
            (com_normed_vectors.unsqueeze(1) * eig_vecs[graph_data["batch"]])
            .sum(
                -1,
            )
            .unsqueeze(1)
        )

        # find projected features for each eig vec
        # [num_graphs, num_feats, 3]
        pos_projected_feats: torch.Tensor = scatter(
            src=torch.where(
                coefs >= 0,
                coefs * feats.unsqueeze(-1),
                torch.zeros_like(feats.unsqueeze(-1)),
            ),
            index=graph_data["batch"],
            dim=0,
            dim_size=int(graph_data["num_graphs"]),
            reduce=self.pool_type,
        )

        neg_projected_feats: torch.Tensor = scatter(
            src=torch.where(
                coefs < 0,
                coefs * feats.unsqueeze(-1),
                torch.zeros_like(feats.unsqueeze(-1)),
            ),
            index=graph_data["batch"],
            dim=0,
            dim_size=int(graph_data["num_graphs"]),
            reduce=self.pool_type,
        )

        # abs for ambiguity in eig vec direction
        pos_projected_feats = torch.abs(pos_projected_feats)
        neg_projected_feats = torch.abs(neg_projected_feats)

        pos_projected_feats = self.lin_down(
            pos_projected_feats.transpose(1, 2),
        ).transpose(1, 2)
        neg_projected_feats = self.lin_down(
            neg_projected_feats.transpose(1, 2),
        ).transpose(1, 2)

        # [num_graphs, 3*num_feats]
        projected_feats = torch.cat(
            [
                pos_projected_feats[:, :, 0] + neg_projected_feats[:, :, 0],
                pos_projected_feats[:, :, 1] + neg_projected_feats[:, :, 1],
                pos_projected_feats[:, :, 2] + neg_projected_feats[:, :, 2],
                pos_projected_feats[:, :, 0] * neg_projected_feats[:, :, 0],
                pos_projected_feats[:, :, 1] * neg_projected_feats[:, :, 1],
                pos_projected_feats[:, :, 2] * neg_projected_feats[:, :, 2],
            ],
            dim=-1,
        )

        return projected_feats

    def get_dot_products(
        self,
        vecs1: torch.Tensor,
        vecs2: torch.Tensor,
    ) -> torch.Tensor:
        # torch.abd because of ambiguity in direction of eigen vecs
        dot_products = torch.stack(
            [
                torch.abs((vecs1[:, 0] * vecs2[:, 0]).sum(-1)),
                torch.abs((vecs1[:, 0] * vecs2[:, 1]).sum(-1)),
                torch.abs((vecs1[:, 0] * vecs2[:, 2]).sum(-1)),
                torch.abs((vecs1[:, 1] * vecs2[:, 0]).sum(-1)),
                torch.abs((vecs1[:, 1] * vecs2[:, 1]).sum(-1)),
                torch.abs((vecs1[:, 1] * vecs2[:, 2]).sum(-1)),
                torch.abs((vecs1[:, 2] * vecs2[:, 0]).sum(-1)),
                torch.abs((vecs1[:, 2] * vecs2[:, 1]).sum(-1)),
                torch.abs((vecs1[:, 2] * vecs2[:, 2]).sum(-1)),
            ],
            dim=-1,
        )
        return dot_products

    def forward(
        self,
        data: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """

        Args:
            data: dict of features.

        Returns: pooled output (bs, 1)

        """

        list_of_feats = []
        for _k, graph_data in data.items():
            interaction_coefs = self.interaction_coefs_mlp(graph_data["node_feats"])

            spatial_eig_dict = self.get_eigen_vectors(
                vectors=graph_data["coordinates"],
                graph_data=graph_data,
            )

            interaction_eig_dict = self.get_eigen_vectors(
                vectors=interaction_coefs * spatial_eig_dict["normed_com_vectors"],
                graph_data=graph_data,
            )

            spatial_feats = self.get_projected_feats(
                feats=graph_data["node_feats"],
                com_normed_vectors=spatial_eig_dict["normed_com_vectors"],
                eig_vecs=spatial_eig_dict["eig_vecs"],
                graph_data=graph_data,
            )

            interaction_feats = self.get_projected_feats(
                feats=graph_data["node_feats"],
                com_normed_vectors=interaction_eig_dict["normed_com_vectors"],
                eig_vecs=interaction_eig_dict["eig_vecs"],
                graph_data=graph_data,
            )
            dot_products = self.get_dot_products(
                vecs1=spatial_eig_dict["eig_vecs"],
                vecs2=interaction_eig_dict["eig_vecs"],
            )

            list_of_feats.append(spatial_feats)
            list_of_feats.append(interaction_feats)
            list_of_feats.append(dot_products)

        pooled_feats: torch.Tensor = self.headMLP2(
            torch.cat(list_of_feats, dim=-1),
        )

        return pooled_feats


class InvariantLigandPocketPoolingHead(torch.nn.Module):
    """
    Class for InvariantLigandPocketPoolingHead
    """

    def __init__(
        self,
        c_hidden: int,
        num_layers_phi: int,
        pool_type: Literal["sum", "mean"],
        pool_from: Literal["nodes", "edges", "nodes_edges"],
        num_tasks: int | None,
        dropout: float | None,
        mlp_activation: str | None,
        mlp_output_activation: str | None,
        output_activation: str | None,
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

        self.headMLP2 = make_mlp(
            c_in=2 * c_hidden,
            c_hidden=c_hidden,
            c_out=num_tasks,
            num_layers=num_layers_phi,
            dropout=dropout,
            mlp_activation=mlp_activation,
            output_activation=output_activation,
        )

    def forward(
        self,
        data: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """

        Args:
            data: dict of features.

        Returns: pooled output (bs, 1)

        """

        list_of_pooled_feats: list[torch.Tensor] = []
        for _k, v in data.items():
            node_feats = v["node_feats"]
            node_feats = self.headMLP1(node_feats)
            pooled_node_feats = scatter(
                src=node_feats,
                index=v["batch"],
                dim=0,
                dim_size=int(v["num_graphs"]),
                reduce=self.pool_type,
            )

            list_of_pooled_feats.append(pooled_node_feats)

        pooled_feats: torch.Tensor = self.headMLP2(
            torch.cat(
                list_of_pooled_feats,
                dim=-1,
            ),
        )
        return pooled_feats
