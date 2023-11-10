from typing import Any, Dict

import torch
from e3nn import o3

from physicsml.models.mace.modules.blocks import (
    InteractionBlock,
    MessageBlock,
    NodeUpdateBlock,
    RadialEmbeddingBlock,
)
from physicsml.models.utils import compute_lengths_and_vectors


class SSF(torch.nn.Module):
    """SSF module"""

    def __init__(
        self,
        irreps: o3.Irreps,
    ) -> None:
        super().__init__()

        num_irreps = irreps.num_irreps
        self.num_scalar = sum(mul for mul, ir in irreps if ir.is_scalar())

        self.gamma = torch.nn.Parameter(torch.ones(num_irreps).unsqueeze(0))
        self.beta = torch.nn.Parameter(torch.zeros(self.num_scalar).unsqueeze(0))
        self.tp = o3.ElementwiseTensorProduct(
            irreps_in1=o3.Irreps(f"{num_irreps}x0e"),
            irreps_in2=irreps,
        )

    def forward(self, equivariant_embedding: torch.Tensor) -> torch.Tensor:
        # scale all features using elementwise tensor product
        equivariant_embedding = self.tp(self.gamma, equivariant_embedding)

        # shift only scalar features
        equivariant_embedding[:, : self.num_scalar] = (
            equivariant_embedding[:, : self.num_scalar] + self.beta
        )

        return equivariant_embedding


class SSFMACE(torch.nn.Module):

    """
    Class for ssf mace model
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
        self.ssf = torch.nn.ModuleList()

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

            self.ssf.append(SSF(hidden_irreps_tmp))

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

            node_feats = self.ssf[idx](node_feats)

            data[f"node_feats_{idx}"] = node_feats

        return data
