import math
from typing import Dict, List, Optional

import torch
from e3nn import o3
from torch_geometric.utils import scatter

from physicsml.models.allegro.modules.channels import MakeWeightedChannels
from physicsml.models.allegro.modules.contract import Contractor
from physicsml.models.allegro.modules.cutoffs import polynomial_cutoff
from physicsml.models.allegro.modules.fc import ScalarMLP, ScalarMLPFunction
from physicsml.models.allegro.modules.linear import Linear
from physicsml.models.allegro.modules.radial import RadialEmbeddingBlock
from physicsml.models.nequip.modules.scale_shift import ScaleShiftBlock
from physicsml.models.utils import compute_lengths_and_vectors


def tp_path_exists(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    ir_out: o3.Irreps,
) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Allegro(torch.nn.Module):
    def __init__(
        self,
        num_node_feats: int,
        num_edge_feats: int,
        num_layers: int,
        max_ell: int,
        num_bessel: int,
        bessel_basis_trainable: bool,
        avg_num_neighbours: float,
        parity: bool,
        num_polynomial_cutoff: int,
        cut_off: float,
        latent_mlp_latent_dimensions: List[int],
        two_body_latent_mlp_latent_dimensions: List[int],
        env_embed_mlp_latent_dimensions: List[int],
        env_embed_multiplicity: int,
        embed_initial_edge: bool,
        per_layer_cutoffs: Optional[List[float]],
        latent_resnet: bool,
        latent_resnet_update_ratios: Optional[List[float]],
        latent_resnet_update_ratios_learnable: bool,
        sparse_mode: Optional[str],
    ):
        super().__init__()
        SCALAR = o3.Irrep("0e")

        self.r_max = cut_off
        self.avg_num_neighbours = avg_num_neighbours

        node_attrs_irreps = o3.Irreps(f"{num_node_feats}x0e")
        edge_feats_irreps = o3.Irreps(f"{num_bessel + num_edge_feats}x0e")

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            trainable=bessel_basis_trainable,
        )

        # spherical harmonics for edge attrs
        parity_int = -1 if parity else 1
        self.spherical_harmonics = o3.SphericalHarmonics(
            o3.Irreps.spherical_harmonics(max_ell, p=parity_int),
            normalize=True,
            normalization="component",
        )

        edge_attrs_irreps = self.spherical_harmonics.irreps_out

        interaction_irreps_in = edge_attrs_irreps  # from sperical harmonics

        self.num_layers = num_layers
        self.latent_resnet = latent_resnet
        self.embed_initial_edge = embed_initial_edge
        self.avg_num_neighbours = avg_num_neighbours
        self.parity = parity
        self.polynomial_cutoff_p = num_polynomial_cutoff

        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor([avg_num_neighbours] * num_layers).rsqrt(),
        )

        self.latents = torch.nn.ModuleList([])
        self.env_embed_mlps = torch.nn.ModuleList([])
        self.tps = torch.nn.ModuleList([])
        self.linears = torch.nn.ModuleList([])
        self.env_linears = torch.nn.ModuleList([])

        assert all(mul == 1 for mul, ir in interaction_irreps_in)
        env_embed_irreps = o3.Irreps([(1, ir) for _, ir in interaction_irreps_in])
        assert (
            env_embed_irreps[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"
        pad_to_alignment = 1
        self._input_pad = (
            int(math.ceil(env_embed_irreps.dim / pad_to_alignment)) * pad_to_alignment
        ) - env_embed_irreps.dim
        self.register_buffer("_zero", torch.zeros(1, 1))

        if self.embed_initial_edge:
            arg_irreps = env_embed_irreps
        else:
            arg_irreps = interaction_irreps_in

        # begin and iterate tps
        tps_irreps = [arg_irreps]

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                ir_out = []
                for _mul, ir in env_embed_irreps:
                    if self.parity:
                        ir_out.append((1, (ir.l, 1)))
                        ir_out.append((1, (ir.l, -1)))
                    else:
                        ir_out.append((1, ir))

                ir_out = o3.Irreps(ir_out)

            if layer_idx == self.num_layers - 1:
                ir_out = o3.Irreps([(1, (0, 1))])

            # Prune impossible paths
            ir_out = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in ir_out
                    if tp_path_exists(arg_irreps, env_embed_irreps, ir)
                ],
            )

            arg_irreps = ir_out
            tps_irreps.append(ir_out)
        # - end build irreps -

        # == Remove unneeded paths ==
        out_irreps = tps_irreps[-1]
        new_tps_irreps = [out_irreps]
        for arg_irreps in reversed(tps_irreps[:-1]):
            new_arg_irreps = []
            for mul, arg_ir in arg_irreps:
                for _, env_ir in env_embed_irreps:
                    if any(i in out_irreps for i in arg_ir * env_ir):
                        new_arg_irreps.append((mul, arg_ir))
                        break
            new_arg_irreps = o3.Irreps(new_arg_irreps)
            new_tps_irreps.append(new_arg_irreps)
            out_irreps = new_arg_irreps

        assert len(new_tps_irreps) == len(tps_irreps)
        tps_irreps = list(reversed(new_tps_irreps))
        del new_tps_irreps

        assert tps_irreps[-1].lmax == 0

        tps_irreps_in = tps_irreps[:-1]
        tps_irreps_out = tps_irreps[1:]
        del tps_irreps

        # Environment builder:
        self._env_weighter = MakeWeightedChannels(
            irreps_in=interaction_irreps_in,
            multiplicity_out=env_embed_multiplicity,
            pad_to_alignment=1,
        )

        self._n_scalar_outs = []

        # # == Build TPs ==
        for layer_idx, (arg_irreps, out_irreps) in enumerate(
            zip(tps_irreps_in, tps_irreps_out),
        ):
            self.env_linears.append(torch.nn.Identity())

            # Make TP
            tmp_i_out: int = 0
            instr = []
            n_scalar_outs: int = 0
            full_out_irreps = []
            for _i_out, (_, ir_out) in enumerate(out_irreps):
                for i_1, (_, ir_1) in enumerate(arg_irreps):
                    for i_2, (_, ir_2) in enumerate(env_embed_irreps):
                        if ir_out in ir_1 * ir_2:
                            if ir_out == SCALAR:
                                n_scalar_outs += 1
                            instr.append((i_1, i_2, tmp_i_out))
                            full_out_irreps.append((env_embed_multiplicity, ir_out))
                            tmp_i_out += 1
            full_out_irreps = o3.Irreps(full_out_irreps)
            self._n_scalar_outs.append(n_scalar_outs)
            assert all(ir == SCALAR for _, ir in full_out_irreps[:n_scalar_outs])

            tp = Contractor(
                irreps_in1=o3.Irreps(
                    [
                        (
                            (
                                env_embed_multiplicity
                                if layer_idx > 0 or self.embed_initial_edge
                                else 1
                            ),
                            ir,
                        )
                        for _, ir in arg_irreps
                    ],
                ),
                irreps_in2=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                ),
                irreps_out=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in full_out_irreps],
                ),
                instructions=instr,
                connection_mode=(
                    "uuu" if layer_idx > 0 or self.embed_initial_edge else "uvv"
                ),
                shared_weights=False,
                has_weight=False,
                pad_to_alignment=1,
                sparse_mode=sparse_mode,
            )
            self.tps.append(tp)
            assert out_irreps[0].ir == SCALAR

            # Make env embed mlp
            generate_n_weights = (
                self._env_weighter.weight_numel
            )  # the weight for the edge embedding
            if layer_idx == 0 and self.embed_initial_edge:
                # also need weights to embed the edge itself
                # this is because the 2 body latent is mixed in with the first layer
                # in terms of code
                generate_n_weights += self._env_weighter.weight_numel

            # the linear acts after the extractor
            self.linears.append(
                Linear(
                    full_out_irreps,
                    [(env_embed_multiplicity, ir) for _, ir in out_irreps],
                    shared_weights=True,
                    internal_weights=True,
                    pad_to_alignment=1,
                ),
            )

            if layer_idx == 0:
                # at the first layer, we have no invariants from previous TPs
                self.latents.append(
                    ScalarMLPFunction(
                        mlp_input_dimension=(
                            2 * node_attrs_irreps.num_irreps
                            + edge_feats_irreps.num_irreps
                        ),
                        mlp_latent_dimensions=two_body_latent_mlp_latent_dimensions,
                        mlp_output_dimension=None,
                        mlp_nonlinearity="silu",
                    ),
                )
            else:
                self.latents.append(
                    ScalarMLPFunction(
                        mlp_input_dimension=(
                            # the embedded latent invariants from the previous layer(s)
                            self.latents[-1].out_features
                            # and the invariants extracted from the last layer's TP:
                            + env_embed_multiplicity * n_scalar_outs
                        ),
                        mlp_latent_dimensions=latent_mlp_latent_dimensions,
                        mlp_output_dimension=None,
                        mlp_nonlinearity="silu",
                    ),
                )

            self.env_embed_mlps.append(
                ScalarMLPFunction(
                    mlp_input_dimension=self.latents[-1].out_features,
                    mlp_latent_dimensions=env_embed_mlp_latent_dimensions,
                    mlp_output_dimension=generate_n_weights,
                    mlp_nonlinearity=None,
                ),
            )

        # Create the final latent layer
        self.final_latent = ScalarMLPFunction(
            mlp_input_dimension=(
                self.latents[-1].out_features + env_embed_multiplicity * n_scalar_outs
            ),
            mlp_latent_dimensions=latent_mlp_latent_dimensions,
            mlp_output_dimension=None,
            mlp_nonlinearity="silu",
        )

        # layer-resnet update weights
        if latent_resnet_update_ratios is None:
            latent_resnet_update_params = torch.zeros(self.num_layers)
        else:
            latent_resnet_update_ratios = torch.as_tensor(latent_resnet_update_ratios)
            assert latent_resnet_update_ratios.min() > 0.0
            assert latent_resnet_update_ratios.min() < 1.0
            latent_resnet_update_params = torch.special.logit(
                latent_resnet_update_ratios,
            )
            # The sigmoid is mostly saturated at ±6, keep it in a reasonable range
            latent_resnet_update_params.clamp_(-6.0, 6.0)
        assert latent_resnet_update_params.shape == (
            num_layers,
        ), f"There must be {num_layers} layer resnet update ratios (layer0:layer1, layer1:layer2)"
        if latent_resnet_update_ratios_learnable:
            self._latent_resnet_update_params = torch.nn.Parameter(
                latent_resnet_update_params,
            )
        else:
            self.register_buffer(
                "_latent_resnet_update_params",
                latent_resnet_update_params,
            )

        # Set the per-layer cutoffs
        if per_layer_cutoffs is None:
            per_layer_cutoffs = torch.full((num_layers + 1,), self.r_max)
        self.register_buffer("per_layer_cutoffs", torch.as_tensor(per_layer_cutoffs))
        assert torch.all(self.per_layer_cutoffs <= self.r_max)
        assert self.per_layer_cutoffs.shape == (
            num_layers + 1,
        ), "Must be one per-layer cutoff for layer 0 and every layer for a total of {num_layers} cutoffs (the first applies to the two body latent, which is 'layer 0')"
        assert (
            self.per_layer_cutoffs[1:] <= self.per_layer_cutoffs[:-1]
        ).all(), "Per-layer cutoffs must be equal or decreasing"
        assert (
            self.per_layer_cutoffs.min() > 0
        ), "Per-layer cutoffs must be >0. To remove higher layers entirely, lower `num_layers`."

        self._latent_dim = self.final_latent.out_features
        self.register_buffer("_zero", torch.tensor(0.0))

        # Set the irreps_out dictionary
        self.irreps_layer_out = o3.Irreps([(self.final_latent.out_features, (0, 1))])

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

        edge_feats = self.radial_embedding(lengths)
        if "edge_attrs" in data:
            data["edge_feats"] = torch.cat([edge_feats, data["edge_attrs"]], dim=-1)
        else:
            data["edge_feats"] = edge_feats
        data["edge_attrs"] = self.spherical_harmonics(vectors)

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

        edge_center = data["edge_index"][0]
        edge_neighbor = data["edge_index"][1]

        # get the edge attributes
        edge_attr = data["edge_attrs"]

        # pad edge_attr
        if self._input_pad > 0:
            edge_attr = torch.cat(
                (
                    edge_attr,
                    self._zero.expand(len(edge_attr), self._input_pad),
                ),
                dim=-1,
            )
        # get the length of each edge
        edge_length = lengths.squeeze()

        # get the number of edges
        num_edges = len(edge_attr)

        # get the edge and node invariants
        edge_invariants = data["edge_feats"]
        node_invariants = data["node_attrs"]

        # initialize some variables
        scalars = self._zero

        # Initialize state
        latents = torch.zeros(
            (num_edges, self._latent_dim),
            dtype=edge_attr.dtype,
            device=edge_attr.device,
        )
        active_edges = torch.arange(num_edges, device=edge_attr.device)

        # For the first layer, we use the input invariants:
        # The center and neighbor invariants and edge invariants
        latent_inputs_to_cat = [
            node_invariants[edge_center],
            node_invariants[edge_neighbor],
            edge_invariants,
        ]
        # The nonscalar features. Initially, the edge data.
        features = edge_attr

        layer_index: int = 0

        # compute the sigmoids vectorized instead of each loop
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid()

        # Vectorized precompute per layer cutoffs
        cutoff_coeffs_all = polynomial_cutoff(
            edge_length,
            self.per_layer_cutoffs,
            p=self.polynomial_cutoff_p,
        )

        for latent, env_embed_mlp, env_linear, tp, linear in zip(
            self.latents,
            self.env_embed_mlps,
            self.env_linears,
            self.tps,
            self.linears,
        ):
            # Determine which edges are still in play
            cutoff_coeffs = cutoff_coeffs_all[layer_index]
            prev_mask = cutoff_coeffs[active_edges] > 0
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)
            # Compute latents
            new_latents = latent(torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask])

            # Apply cutoff, which propagates through to everything else
            new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents

            if self.latent_resnet and layer_index > 0:
                this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
                coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
                coefficient_new = this_layer_update_coeff * coefficient_old
                latents = torch.index_add(
                    coefficient_old * latents,
                    0,
                    active_edges,
                    coefficient_new * new_latents,
                )
            else:
                latents = torch.index_copy(
                    latents,
                    0,
                    active_edges,
                    new_latents,
                )

            # From the latents, compute the weights for active edges:
            weights = env_embed_mlp(latents[active_edges])
            w_index: int = 0

            if self.embed_initial_edge and layer_index == 0:
                # embed initial edge
                env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
                w_index += self._env_weighter.weight_numel
                features = self._env_weighter(
                    features[prev_mask],
                    env_w,
                )  # features is edge_attr
            else:
                # just take the previous features that we still need
                features = features[prev_mask]

            # Extract weights for the environment builder
            env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
            w_index += self._env_weighter.weight_numel

            local_env_per_edge = scatter(
                self._env_weighter(edge_attr[active_edges], env_w),
                edge_center[active_edges],
                dim=0,
            )

            if self.env_sum_normalizations.ndim < 2:
                # it's a scalar per layer
                norm_const = self.env_sum_normalizations[layer_index]
            else:
                # it's per type
                # get shape [N_atom, 1] for broadcasting
                norm_const = self.env_sum_normalizations[
                    layer_index,
                    data["batch"],
                ].unsqueeze(-1)

            local_env_per_edge = local_env_per_edge * norm_const
            local_env_per_edge = env_linear(local_env_per_edge)
            # Copy to get per-edge
            # Large allocation, but no better way to do this:
            local_env_per_edge = local_env_per_edge[edge_center[active_edges]]

            # Now do the TP
            # recursively tp current features with the environment embeddings
            features = tp(features, local_env_per_edge)

            # Get invariants
            # features has shape [z][mul][k]
            # we know scalars are first
            scalars = features[:, :, : self._n_scalar_outs[layer_index]].reshape(
                features.shape[0],
                -1,
            )

            # do the linear
            features = linear(features)

            # For layer2+, use the previous latents and scalars
            # This makes it deep
            latent_inputs_to_cat = [
                latents[active_edges],
                scalars,
            ]

            # increment counter
            layer_index += 1

        # - final layer -
        cutoff_coeffs = cutoff_coeffs_all[layer_index]
        prev_mask = cutoff_coeffs[active_edges] > 0
        active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)
        new_latents = self.final_latent(
            torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask],
        )
        new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
        if self.latent_resnet:
            this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
            coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
            coefficient_new = this_layer_update_coeff * coefficient_old
            latents = torch.index_add(
                coefficient_old * latents,
                0,
                active_edges,
                coefficient_new * new_latents,
            )
        else:
            latents = torch.index_copy(latents, 0, active_edges, new_latents)

        # final latents
        data["edge_feats"] = latents

        return data


class ReadoutHead(torch.nn.Module):
    def __init__(
        self,
        irrreps_in: o3.Irreps,
        mlp_irreps: o3.Irreps,
        mlp_latent_dimensions: List[int],
        num_tasks: int,
        avg_num_neighbours: float,
        scaling_std: float,
        scaling_mean: float,
    ) -> None:
        super().__init__()

        self.scalar_mlp = ScalarMLP(
            irreps_in=irrreps_in,
            irreps_out=mlp_irreps,
            mlp_output_dimension=num_tasks,
            mlp_latent_dimensions=mlp_latent_dimensions,
        )

        self.inv_factor = 1.0 / math.sqrt(avg_num_neighbours)
        self.scale_shift = ScaleShiftBlock(scale=scaling_std, shift=scaling_mean)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        edge_feats = self.scalar_mlp(data)

        # Sum over edges in graph to get graph embeddings
        node_feats = scatter(
            src=edge_feats,
            index=data["edge_index"][0],
            dim=0,
            dim_size=int(data["num_nodes"]),
        )

        node_feats = node_feats * self.inv_factor
        node_feats = self.scale_shift(node_feats)

        return node_feats


class PooledReadoutHead(torch.nn.Module):
    def __init__(
        self,
        irrreps_in: o3.Irreps,
        mlp_irreps: o3.Irreps,
        mlp_latent_dimensions: List[int],
        num_tasks: int,
        avg_num_neighbours: float,
        scaling_std: float,
        scaling_mean: float,
    ) -> None:
        super().__init__()

        self.scalar_mlp = ScalarMLP(
            irreps_in=irrreps_in,
            irreps_out=mlp_irreps,
            mlp_output_dimension=num_tasks,
            mlp_latent_dimensions=mlp_latent_dimensions,
        )

        self.inv_factor = 1.0 / math.sqrt(avg_num_neighbours)
        self.scale_shift = ScaleShiftBlock(scale=scaling_std, shift=scaling_mean)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        edge_feats = self.scalar_mlp(data)

        # Sum over edges in graph to get graph embeddings
        node_feats = scatter(
            src=edge_feats,
            index=data["edge_index"][0],
            dim=0,
            dim_size=int(data["num_nodes"]),
        )

        node_feats = node_feats * self.inv_factor
        node_feats = self.scale_shift(node_feats)

        pooled_feats = scatter(
            src=node_feats,
            index=data["batch"],
            dim=0,
            dim_size=int(data["num_graphs"]),
        )

        return pooled_feats
