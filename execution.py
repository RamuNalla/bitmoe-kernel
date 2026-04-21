import torch
from bitmoe.moe_block import SparseMoEBlock
from bitmoe.experts.bitlinear import TernaryQuantizeSTE

def execute_day_3():
    if not torch.cuda.is_available():
        print("CRITICAL: Triton kernels require an NVIDIA GPU. Please run this in Google Colab.")
        return

    print("Initializing MoE Block on CUDA...")
    torch.manual_seed(42)
    d_model, hidden_dim, num_experts, top_k = 128, 512, 8, 2
    batch_size, seq_len = 2, 512
    
    model = SparseMoEBlock(d_model, hidden_dim, num_experts, top_k).cuda().half()
    
    # We must explicitly quantize the weights first so both runs use the exact same {-1, 0, 1} matrix
    with torch.no_grad():
        for expert in model.experts:
            expert.w1.weight.copy_(TernaryQuantizeSTE.apply(expert.w1.weight))
            # Convert to int8 format to simulate the low-bandwidth memory read
            expert.w1.weight.data = expert.w1.weight.data.to(torch.int8)

    dummy_input = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float16)

    print("Executing standard PyTorch Dispatch...")
    pytorch_out, _ = model(dummy_input, use_triton=False)

    print("Executing custom Triton Fused Dispatch...")
    triton_out, _ = model(dummy_input, use_triton=True)

    print("\n--- Validation ---")
    # We use allclose with a small tolerance because CUDA float16 math ordering can cause minor rounding differences
    is_matching = torch.allclose(pytorch_out, triton_out, atol=1e-2, rtol=1e-2)
    
    if is_matching:
        print("SUCCESS! The Triton Kernel output perfectly matches the PyTorch mathematical baseline.")
        print("Max absolute difference:", torch.max(torch.abs(pytorch_out - triton_out)).item())
    else:
        print("FAILURE: The matrices do not match. Check pointer logic.")

if __name__ == "__main__":
    execute_day_3()