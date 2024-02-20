from typing import Any, Dict

import torch
from e3nn import o3
from torch_geometric.utils import scatter

from physicsml.models.mace.modules.blocks import (
    InteractionBlock,
    MessageBlock,
    NodeUpdateBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from physicsml.models.utils import compute_lengths_and_vectors


class MACE(torch.nn.Module):

    """
    Class for mace model
    """

    def __init__(
        self,
        cut_off: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        num_node_feats: int,
        num_edge_feats: int,
        hidden_irreps: str,
        avg_num_neighbours: float,
        correlation: int,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.r_max = cut_off

        # defining irreps
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        num_0e_features = self.hidden_irreps.count(o3.Irrep(0, 1))

        node_attr_irreps = o3.Irreps(f"{num_node_feats}x0e")
        node_feats_irreps = o3.Irreps(f"{num_0e_features}x0e")
        edge_feats_irreps = o3.Irreps(f"{num_bessel + num_edge_feats}x0e")

        # spherical harmonics for edge attrs
        self.spherical_harmonics = o3.SphericalHarmonics(
            o3.Irreps.spherical_harmonics(max_ell),
            normalize=True,
            normalization="component",
        )

        # Embeddings
        self.node_embedding = o3.Linear(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )

        # Interactions, tensor products, and readouts
        interaction_irreps = (
            (self.spherical_harmonics.irreps_out * num_0e_features)
            .sort()
            .irreps.simplify()
        )

        # initialised Module Lists
        self.interactions = torch.nn.ModuleList()
        self.messages = torch.nn.ModuleList()
        self.node_updates = torch.nn.ModuleList()

        # middle blocks
        self.out_irreps = []
        for i in range(num_interactions):
            if i == 0:
                node_feats_irreps_tmp = node_feats_irreps
                hidden_irreps_tmp = self.hidden_irreps
                mix_with_node_attrs = True
                residual_connection = False
            elif i == num_interactions - 1:
                node_feats_irreps_tmp = self.hidden_irreps
                hidden_irreps_tmp = o3.Irreps(
                    f"{num_0e_features}x0e",
                )  # Select only scalars for last layer
                mix_with_node_attrs = False
                residual_connection = True
            else:
                node_feats_irreps_tmp = self.hidden_irreps
                hidden_irreps_tmp = self.hidden_irreps
                mix_with_node_attrs = False
                residual_connection = True

            self.interactions.append(
                InteractionBlock(
                    node_feats_irreps=node_feats_irreps_tmp,
                    node_attrs_irreps=node_attr_irreps,
                    edge_attrs_irreps=self.spherical_harmonics.irreps_out,
                    edge_feats_irreps=edge_feats_irreps,
                    interaction_irreps=interaction_irreps,
                    avg_num_neighbours=avg_num_neighbours,
                    mix_with_node_attrs=mix_with_node_attrs,
                ),
            )

            self.messages.append(
                MessageBlock(
                    interaction_irreps=interaction_irreps,
                    node_attrs_irreps=node_attr_irreps,
                    hidden_irreps=hidden_irreps_tmp,
                    correlation=correlation,
                ),
            )

            self.node_updates.append(
                NodeUpdateBlock(
                    node_feats_irreps=node_feats_irreps_tmp,
                    node_attrs_irreps=node_attr_irreps,
                    hidden_irreps=hidden_irreps_tmp,
                    residual_connection=residual_connection,
                ),
            )

            self.out_irreps.append(hidden_irreps_tmp)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "cell" in data:
            cell = data["cell"]
            cell_shift_vector = data["cell_shift_vector"]
        else:
            cell = None
            cell_shift_vector = None

        lengths, vectors = compute_lengths_and_vectors(
            positions=data["coordinates"],
            edge_index=data["edge_index"],
            cell=cell,
            cell_shift_vector=cell_shift_vector,
        )

        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        if "edge_attrs" in data:
            edge_feats = torch.cat([edge_feats, data["edge_attrs"]], dim=-1)

        for idx, (interaction, message, node_update) in enumerate(
            zip(self.interactions, self.messages, self.node_updates),
        ):
            a_i = interaction(
                node_feats=node_feats,
                node_attrs=data["node_attrs"],
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )

            m_i = message(a_i=a_i, node_attrs=data["node_attrs"])

            node_feats = node_update(
                m_i=m_i,
                node_feats=node_feats,
                node_attrs=data["node_attrs"],
            )

            data[f"node_feats_{idx}"] = node_feats

        return data


class ReadoutHead(torch.nn.Module):
    def __init__(
        self,
        list_in_irreps: o3.Irreps,
        mlp_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
        scaling_std: float,
        scaling_mean: float,
    ) -> None:
        super().__init__()

        self.readouts = torch.nn.ModuleList()

        for idx, in_irreps in enumerate(list_in_irreps):
            if idx < len(list_in_irreps) - 1:
                self.readouts.append(
                    o3.Linear(irreps_in=in_irreps, irreps_out=out_irreps),
                )
            else:
                self.readouts.append(
                    NonLinearReadoutBlock(in_irreps, mlp_irreps, out_irreps),
                )

        self.scale_shift = ScaleShiftBlock(scale=scaling_std, shift=scaling_mean)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_feats_list = []
        for idx, readout in enumerate(self.readouts):
            node_feats = readout(data[f"node_feats_{idx}"])
            node_feats_list.append(node_feats)

        # Sum all interaction node feats
        node_feats = torch.sum(torch.stack(node_feats_list, dim=0), dim=0)

        node_feats_out: torch.Tensor = self.scale_shift(node_feats)

        return node_feats_out


class PooledReadoutHead(torch.nn.Module):
    def __init__(
        self,
        list_in_irreps: o3.Irreps,
        mlp_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
        scaling_std: float,
        scaling_mean: float,
    ) -> None:
        super().__init__()

        self.readouts = torch.nn.ModuleList()

        for idx, in_irreps in enumerate(list_in_irreps):
            if idx < len(list_in_irreps) - 1:
                self.readouts.append(
                    o3.Linear(irreps_in=in_irreps, irreps_out=out_irreps),
                )
            else:
                self.readouts.append(
                    NonLinearReadoutBlock(in_irreps, mlp_irreps, out_irreps),
                )

        self.scale_shift = ScaleShiftBlock(scale=scaling_std, shift=scaling_mean)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_feats_list = []
        for idx, readout in enumerate(self.readouts):
            node_feats = readout(data[f"node_feats_{idx}"])
            node_feats_list.append(node_feats)

        # Sum all interaction node feats
        node_feats = torch.sum(torch.stack(node_feats_list, dim=0), dim=0)
        node_feats = self.scale_shift(node_feats)

        # Sum over nodes in graph to get graph outputs
        pooled_feats: torch.Tensor = scatter(
            src=node_feats,
            index=data["batch"],
            dim=0,
            dim_size=int(data["num_graphs"]),
        )

        return pooled_feats
