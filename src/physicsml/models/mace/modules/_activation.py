from typing import List, Optional

import torch
from e3nn import o3
from e3nn.math import normalize2mom


class Activation(torch.nn.Module):
    r"""Scalar activation function.

    Odd scalar inputs require activation functions with a defined parity (odd or even).

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    acts : list of function or None
        list of activation functions, `None` if non-scalar or identity

    Examples
    --------

    >>> a = Activation("256x0o", [torch.abs])
    >>> a.irreps_out
    256x0e

    >>> a = Activation("256x0o+16x1e", [None, None])
    >>> a.irreps_out
    256x0o+16x1e
    """

    def __init__(self, irreps_in: o3.Irreps, acts: List[Optional[torch.nn.Module]]):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        if len(irreps_in) != len(acts):
            raise ValueError(
                f"Irreps in and number of activation functions does not match: {len(acts), (irreps_in, acts)}",
            )

        # normalize the second moment
        acts = [normalize2mom(act) if act is not None else None for act in acts]

        from e3nn.util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(
                        "Activation: cannot apply an activation function to a non-scalar input.",
                    )

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither "
                        "even nor odd.",
                    )
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(acts)  # type: ignore
        assert len(self.irreps_in) == len(self.acts)

        self.ir_dims: List[int] = [ir.dim for _, ir in self.irreps_in]

    def __repr__(self) -> str:
        acts = "".join(["x" if a is not None else " " for a in self.acts])
        return f"{self.__class__.__name__} [{acts}] ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(...)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape the same shape as the input
        """
        # - PROFILER - with torch.autograd.profiler.record_function(repr(self)):
        output = []
        index = 0

        for i, act in enumerate(self.acts):
            ir_dim = self.ir_dims[i]
            mul, ir = self.irreps_in[i]

            if act is not None:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * ir_dim))
            index += mul * ir_dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            out: torch.Tensor = output[0]
            return out
        else:
            return torch.zeros_like(features)
