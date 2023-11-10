from typing import Any, Dict, List, Optional

import torch
from e3nn import o3
from e3nn.nn import BatchNorm

from physicsml.models.nequip.modules._activation import Activation
from physicsml.models.nequip.modules.convnet_layer import ConvNetLayer
from physicsml.models.nequip.modules.radial import RadialEmbeddingBlock
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


class AdapterNequip(torch.nn.Module):
    def __init__(
        self,
        cut_off: float,
        num_layers: int,
        max_ell: int,
        parity: bool,
        num_features: int,
        ratio_adapter_down: int,
        initial_s: float,
        num_bessel: int,
        bessel_basis_trainable: bool,
        num_polynomial_cutoff: int,
        self_connection: bool,
        resnet: bool,
        num_node_feats: int,
        num_edge_feats: int,
        avg_num_neighbours: float,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.r_max = cut_off

        node_attrs_irreps = o3.Irreps(f"{num_node_feats}x0e")
        node_feats_irreps = o3.Irreps(f"{num_features}x0e")
        edge_feats_irreps = o3.Irreps(f"{num_bessel + num_edge_feats}x0e")

        # spherical harmonics for edge attrs
        parity_int = -1 if parity else 1
        self.spherical_harmonics = o3.SphericalHarmonics(
            o3.Irreps.spherical_harmonics(max_ell, p=parity_int),
            normalize=True,
            normalization="component",
        )

        edge_attrs_irreps = self.spherical_harmonics.irreps_out

        # Embeddings
        self.node_embedding = o3.Linear(
            irreps_in=node_attrs_irreps,
            irreps_out=node_feats_irreps,
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            trainable=bessel_basis_trainable,
        )

        hidden_irreps_list = [
            (num_features, f"{irrep_l}e") for irrep_l in range(max_ell + 1)
        ]
        if parity:
            hidden_irreps_list += [
                (num_features, f"{irrep_l}o") for irrep_l in range(max_ell + 1)
            ]
        hidden_irreps = o3.Irreps(hidden_irreps_list).sort().irreps.simplify()

        interaction_irreps_in = node_feats_irreps

        self.out_irreps = []
        self.adapters_before = torch.nn.ModuleList()
        self.adapters_after = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.s_params_before = torch.nn.ParameterList()
        self.s_params_after = torch.nn.ParameterList()

        self.conv_layers = torch.nn.ModuleList()
        for _i in range(num_layers):
            conv_layer = ConvNetLayer(
                interaction_irreps_in=interaction_irreps_in,
                node_attrs_irreps=node_attrs_irreps,
                edge_feats_irreps=edge_feats_irreps,
                edge_attrs_irreps=edge_attrs_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbours=avg_num_neighbours,
                self_connection=self_connection,
                num_layers=num_layers,
                resnet=resnet,
            )

            self.conv_layers.append(conv_layer)

            new_interaction_irreps_in_list = []
            for mul, irr in hidden_irreps:
                if irr in conv_layer.conv_irreps_out:
                    new_interaction_irreps_in_list.append((mul, irr))

            interaction_irreps_in_new = o3.Irreps(new_interaction_irreps_in_list)

            self.batch_norms.append(BatchNorm(interaction_irreps_in_new))

            self.adapters_before.append(
                Adapter(
                    irreps_in=interaction_irreps_in,
                    irreps_out=interaction_irreps_in_new,
                    ratio_adapter_down=ratio_adapter_down,
                ),
            )
            self.s_params_before.append(torch.nn.Parameter(torch.tensor(initial_s)))
            self.adapters_after.append(
                Adapter(
                    irreps_in=interaction_irreps_in_new,
                    irreps_out=interaction_irreps_in_new,
                    ratio_adapter_down=ratio_adapter_down,
                ),
            )
            self.s_params_after.append(torch.nn.Parameter(torch.tensor(initial_s)))

            interaction_irreps_in = interaction_irreps_in_new
            self.out_irreps.append(interaction_irreps_in)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Embeddings
        data["node_feats"] = self.node_embedding(data["node_attrs"])

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

        edge_feats = self.radial_embedding(lengths)
        if "edge_attrs" in data:
            data["edge_feats"] = torch.cat([edge_feats, data["edge_attrs"]], dim=-1)
        else:
            data["edge_feats"] = edge_feats
        data["edge_attrs"] = self.spherical_harmonics(vectors)

        for idx, conv_layer in enumerate(self.conv_layers):
            ada_node_feats_before = self.adapters_before[idx](data["node_feats"])
            data = conv_layer(data)
            ada_node_feats_after = self.adapters_after[idx](data["node_feats"])

            data["node_feats"] = (
                self.batch_norms[idx](data["node_feats"])
                + self.s_params_before[idx] * ada_node_feats_before
                + self.s_params_after[idx] * ada_node_feats_after
            )

        return data
