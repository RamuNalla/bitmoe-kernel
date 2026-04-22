#!/usr/bin/env python3
"""
Mathematical validation for BitMoE-158:

1) Ternary STE forward matches explicit round(clamp(w/γ))·γ on CPU.
2) Fused Triton MoE w1 path matches a slow FP32 PyTorch reference on CUDA
   (same sort/top-k semantics as fused_bitmoe_forward inputs).

Run from repo root:
  pip install -r requirements.txt
  PYTHONPATH=. python scripts/validate_forward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path so `import bitmoe` works without pip install -e .
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F

from bitmoe.experts.bitlinear import TernaryQuantizeSTE
from bitmoe.kernels.fused_dispatch import fused_bitmoe_forward


def test_ste_matches_manual() -> None:
    torch.manual_seed(0)
    w = torch.randn(48, 96, dtype=torch.float32)
    scale = w.abs().mean().clamp(min=1e-8)
    q = torch.round(torch.clamp(w / scale, -1.0, 1.0))
    expected = q * scale
    got = TernaryQuantizeSTE.apply(w)
    if not torch.allclose(got, expected, rtol=0, atol=1e-6):
        raise AssertionError(
            f"STE forward mismatch: max |diff| = {(got - expected).abs().max().item()}"
        )
    print("[ok] TernaryQuantizeSTE forward matches manual ternary dequantization.")


def reference_fused_w1(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    router_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Slow reference: same math as the fused kernel (int8 W → fp16 for matmul, FP32 acc, then FP16 out)."""
    M, K = x.shape
    E, K_w, N = w_int8.shape
    if E != num_experts or K_w != K:
        raise ValueError("Shape mismatch between x, w, num_experts")

    x_f = x.float()
    w_f = w_int8.to(dtype=torch.float16).float()
    out = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    top_k = router_indices.shape[1]

    for t in range(M):
        for k in range(top_k):
            e = int(router_indices[t, k].item())
            rw = float(routing_weights[t, k].item())
            out[t] += rw * (x_f[t] @ w_f[e])

    return out.half()


def test_fused_vs_reference() -> None:
    if not torch.cuda.is_available():
        print("[skip] Fused vs reference: no CUDA device.")
        return

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    M, K, N, E, top_k = 64, 48, 32, 4, 2
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randint(-1, 2, (E, K, N), device="cuda", dtype=torch.int8)

    logits = torch.randn(M, E, device="cuda", dtype=torch.float16)
    vals, selected = torch.topk(logits, top_k, dim=-1)
    routing_weights = F.softmax(vals, dim=-1)

    ref = reference_fused_w1(x, w, selected, routing_weights, E)
    triton_out = fused_bitmoe_forward(x, w, selected, routing_weights, E)

    diff = (ref.float() - triton_out.float()).abs()
    max_err = diff.max().item()
    # FP16 outputs + atomic adds: allow modest tolerance
    tol = 2e-2
    if max_err > tol:
        raise AssertionError(f"fused vs reference max |diff| = {max_err} (tol {tol})")
    print(f"[ok] fused_bitmoe_forward matches reference (max |diff| = {max_err:.6g}).")


def main() -> None:
    test_ste_matches_manual()
    test_fused_vs_reference()
    print("All validation checks passed.")


if __name__ == "__main__":
    main()
