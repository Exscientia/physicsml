from typing import Any, Dict, List, Optional

import torch
from e3nn import o3
from e3nn.nn import BatchNorm

from physicsml.models.mace.modules._activation import Activation
from physicsml.models.mace.modules.blocks import (
    InteractionBlock,
    MessageBlock,
    NodeUpdateBlock,
    RadialEmbeddingBlock,
)
from physicsml.models.utils import compute_lengths_and_vectors


class Adapter(torch.nn.Module):
    """Adapter module"""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        ratio_adapter_down: int,
    ) -> None:
        super().__init__()

        irreps_down = o3.Irreps(
            [(ir[0] // ratio_adapter_down, ir[1]) for ir in irreps_in],
        )
        acts: List[Optional[torch.nn.Module]] = []
        for _mul, ir in irreps_down:
            if ir.is_scalar():
                acts.append(torch.nn.SiLU())
            else:
                acts.append(None)

        self.linear_down = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_down)
        self.non_linearity = Activation(irreps_in=irreps_down, acts=acts)
        self.linear_up = o3.Linear(irreps_in=irreps_down, irreps_out=irreps_out)
        self.bn = BatchNorm(irreps_out)

    def forward(self, equivariant_embedding: torch.Tensor) -> torch.Tensor:
        equivariant_embedding = self.linear_down(equivariant_embedding)
        equivariant_embedding = self.non_linearity(equivariant_embedding)
        equivariant_embedding = self.linear_up(equivariant_embedding)
        equivariant_embedding = self.bn(equivariant_embedding)
        return equivariant_embedding


class AdapterMACE(torch.nn.Module):

    """
    Class for adapter mace model
    """

    def __init__(
        self,
        cut_off: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        ratio_adapter_down: int,
        initial_s: float,
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
        self.adapters_before = torch.nn.ModuleList()
        self.adapters_after = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.s_params_before = torch.nn.ParameterList()
        self.s_params_after = torch.nn.ParameterList()

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
            self.batch_norms.append(BatchNorm(hidden_irreps_tmp))

            self.adapters_before.append(
                Adapter(
                    irreps_in=node_feats_irreps_tmp,
                    irreps_out=hidden_irreps_tmp,
                    ratio_adapter_down=ratio_adapter_down,
                ),
            )
            self.s_params_before.append(torch.nn.Parameter(torch.tensor(initial_s)))
            self.adapters_after.append(
                Adapter(
                    irreps_in=hidden_irreps_tmp,
                    irreps_out=hidden_irreps_tmp,
                    ratio_adapter_down=ratio_adapter_down,
                ),
            )
            self.s_params_after.append(torch.nn.Parameter(torch.tensor(initial_s)))

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
            ada_node_feats_before = self.adapters_before[idx](node_feats)
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

            ada_node_feats_after = self.adapters_after[idx](node_feats)

            node_feats = (
                self.batch_norms[idx](node_feats)
                + self.s_params_before[idx] * ada_node_feats_before
                + self.s_params_after[idx] * ada_node_feats_after
            )

            data[f"node_feats_{idx}"] = node_feats

        return data
