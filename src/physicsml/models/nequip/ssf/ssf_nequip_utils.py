from typing import Any, Dict

import torch
from e3nn import o3

from physicsml.models.nequip.modules.convnet_layer import ConvNetLayer
from physicsml.models.nequip.modules.radial import RadialEmbeddingBlock
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


class SSFNequip(torch.nn.Module):
    def __init__(
        self,
        cut_off: float,
        num_layers: int,
        max_ell: int,
        parity: bool,
        num_features: int,
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
        self.ssf = torch.nn.ModuleList()

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

            interaction_irreps_in = o3.Irreps(new_interaction_irreps_in_list)

            self.ssf.append(SSF(interaction_irreps_in))

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
            data = conv_layer(data)
            data["node_feats"] = self.ssf[idx](data["node_feats"])

        return data
