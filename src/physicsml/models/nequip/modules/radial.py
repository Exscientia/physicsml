import torch


class BesselBasis(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        trainable: bool = False,
    ) -> None:
        super().__init__()

        bessel_weights = torch.pi * torch.arange(1, num_basis + 1)

        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.r_max = r_max
        # should have sqrt, but nequip original code does not have it
        self.prefactor = 2.0 / r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        bessel_funcs = self.prefactor * (
            torch.sin(self.bessel_weights * x / self.r_max) / x
        )  # [..., num_basis]
        return bessel_funcs


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
        self.bessel_fn = BesselBasis(
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
