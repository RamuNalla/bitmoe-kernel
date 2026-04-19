import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardExpert(nn.Module):
    """A standard high-precision FP16 Feed-Forward Network."""
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard SwiGLU activation pattern
        return self.w2(F.silu(self.w1(x)) * self.w3(x))