from math import sqrt

import torch


class BesselBasis(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        trainable: bool = False,
    ) -> None:
        super().__init__()

        bessel_weights = torch.pi * torch.arange(1, num_basis + 1) / r_max

        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.r_max = r_max
        self.prefactor = sqrt(2.0 / r_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        bessel_funcs = self.prefactor * (
            torch.sin(self.bessel_weights * x) / x
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
