import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitLinear

class BitExpert(nn.Module):
    """A 1.58-bit Feed-Forward Network."""
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        # Replacing standard linear layers with Ternary BitLinear layers
        self.w1 = BitLinear(d_model, hidden_dim, bias=False)
        self.w2 = BitLinear(hidden_dim, d_model, bias=False)
        self.w3 = BitLinear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))