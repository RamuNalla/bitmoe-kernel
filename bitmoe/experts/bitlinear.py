import torch
import torch.nn as nn
import torch.nn.functional as F

class TernaryQuantizeSTE(torch.autograd.Function):
    """
    Bypasses the zero-gradient problem of discrete step functions during Backpropagation.
    """
    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        # 1. Calculate the scaling factor (gamma)
        scale = weight.abs().mean().clamp(min=1e-8)
        
        # 2. Scale, clamp to [-1, 1], and round to nearest integer {-1, 0, 1}
        quantized = torch.round(torch.clamp(weight / scale, -1.0, 1.0))
        
        # 3. De-quantize to maintain mathematical magnitude consistency
        return quantized * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # STE: Pass the gradient straight through to the latent FP32 weights
        return grad_output

class BitLinear(nn.Module):
    """A drop-in replacement for nn.Linear using 1.58-bit precision."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the custom STE quantization on-the-fly during forward pass
        quantized_weight = TernaryQuantizeSTE.apply(self.weight)
        return F.linear(x, quantized_weight, self.bias)