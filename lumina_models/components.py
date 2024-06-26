import warnings

import torch
import torch.nn as nn


try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ImportError:
    warnings.warn("Cannot import apex RMSNorm, trying flash_attn RMSNorm")
try:
    from flash_attn.ops.triton.layer_norm import RMSNorm
except:
    try:
        from flash_attn.ops.rms_norm import RMSNorm
    except ImportError:
        warnings.warn("Cannot import flash_attn RMSNorm, falling back to PyTorch RMSNorm")

        class RMSNorm(torch.nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6):
                """
                Initialize the RMSNorm normalization layer.

                Args:
                    dim (int): The dimension of the input tensor.
                    eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

                Attributes:
                    eps (float): A small value added to the denominator for numerical stability.
                    weight (nn.Parameter): Learnable scaling parameter.

                """
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def _norm(self, x):
                """
                Apply the RMSNorm normalization to the input tensor.

                Args:
                    x (torch.Tensor): The input tensor.

                Returns:
                    torch.Tensor: The normalized tensor.

                """
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

            def forward(self, x):
                """
                Forward pass through the RMSNorm layer.

                Args:
                    x (torch.Tensor): The input tensor.

                Returns:
                    torch.Tensor: The output tensor after applying RMSNorm.

                """
                output = self._norm(x.float()).type_as(x)
                return output * self.weight
