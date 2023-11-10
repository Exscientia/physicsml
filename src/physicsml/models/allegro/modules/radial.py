import torch


class BesselBasis(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        trainable: bool = True,
    ) -> None:
        super().__init__()

        bessel_weights = torch.pi * torch.arange(1, num_basis + 1)

        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.r_max = r_max
        self.num_basis = num_basis
        # should have sqrt, but nequip original code does not have it
        self.prefactor = 2.0 / r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        bessel_funcs = self.prefactor * (
            torch.sin(self.bessel_weights * x / self.r_max) / x
        )  # [..., num_basis]
        return bessel_funcs


class NormalizedBasis(torch.nn.Module):
    def __init__(
        self,
        r_max: float = 6.0,
        num_basis: int = 8,
        r_min: float = 0.0,
        trainable: bool = True,
        n: int = 4000,
        norm_basis_mean_shift: bool = True,
    ):
        super().__init__()
        self.basis = BesselBasis(r_max=r_max, num_basis=num_basis, trainable=trainable)
        self.r_min = r_min
        self.r_max = r_max
        assert self.r_min >= 0.0
        assert self.r_max > r_min
        self.n = n

        self.num_basis = self.basis.num_basis

        # Uniform distribution on [r_min, r_max)
        rs = torch.linspace(r_min, r_max, n + 1)[1:]
        rs = rs.unsqueeze(-1)
        bs = self.basis(rs)
        assert bs.ndim == 2 and len(bs) == n
        if norm_basis_mean_shift:
            basis_std, basis_mean = torch.std_mean(bs, dim=0)
        else:
            basis_std = bs.square().mean().sqrt()
            basis_mean = torch.as_tensor(
                0.0,
                device=basis_std.device,
                dtype=basis_std.dtype,
            )

        self._mean = torch.nn.Parameter(basis_mean, requires_grad=False)
        self._inv_std = torch.nn.Parameter(
            torch.reciprocal(basis_std),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.basis(x) - self._mean) * self._inv_std


class PolynomialCutoff(torch.nn.Module):
    def __init__(self, r_max: float, p: int = 6) -> None:
        super().__init__()

        self.r_max = r_max
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0)
            * torch.pow(x / self.r_max, self.p)
            + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
            - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )

        poly: torch.Tensor = envelope * (x < self.r_max)
        return poly


class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.bessel_fn = NormalizedBasis(
            r_max=r_max,
            num_basis=num_bessel,
            trainable=trainable,
        )
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
    ) -> torch.Tensor:
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]

        output: torch.Tensor = bessel * cutoff  # [n_edges, n_basis]
        return output
