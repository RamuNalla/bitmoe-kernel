import torch
import triton
import triton.language as tl

@triton.jit
def _fused_158b_moe_kernel(
    # Pointers to matrices
    X_ptr, W_ptr, Out_ptr,
    # Pointers to routing data
    expert_indices_ptr, routing_weights_ptr, sorted_token_indices_ptr, expert_offsets_ptr,
    # Matrix dimensions
    M, K, N,
    # Strides for X (Tokens, Hidden_Dim)
    stride_xm, stride_xk,
    # Strides for W (Expert, Hidden_Dim, Out_Dim)
    stride_we, stride_wk, stride_wn,
    # Strides for Out (Tokens, Out_Dim)
    stride_outm, stride_outn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fuses the sparse token retrieval and the ternary matrix multiplication.
    """
    # 1. Identify which Expert block this thread block is processing
    expert_id = tl.program_id(1)
    
    # 2. Find the start and end of the tokens assigned to this expert
    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
    num_tokens_for_expert = expert_end - expert_start
    
    if num_tokens_for_expert == 0:
        return # Skip if this expert got no tokens (handles load imbalance safely)

    # 3. Identify the specific chunk of tokens this thread block will handle
    pid_m = tl.program_id(0)
    
    # 4. Create block pointers for the M (Tokens), N (Output Dim), and K (Hidden Dim) axes
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    
    # Ensure we don't read past the number of tokens assigned to this expert
    mask_m = offs_am < num_tokens_for_expert
    
    # 5. Fetch the original token indices for this specific block
    token_idx_ptrs = sorted_token_indices_ptr + expert_start + offs_am
    original_token_indices = tl.load(token_idx_ptrs, mask=mask_m, other=0)
    
    # 6. Fetch the routing weights for these specific tokens
    weight_ptrs = routing_weights_ptr + expert_start + offs_am
    router_weights = tl.load(weight_ptrs, mask=mask_m, other=0.0)

    # 7. Calculate memory offsets for X and W
    x_ptrs = X_ptr + (original_token_indices[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (expert_id * stride_we + offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    
    # 8. Initialize the accumulator for the dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 9. Block-Tiled Matrix Multiplication Loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load X in FP16
        x = tl.load(x_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        # Load W (The 1.58b {-1, 0, 1} weights, stored as int8 to save bandwidth)
        w_int8 = tl.load(w_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0)
        
        # Convert quantized weights to float16 on the SRAM for the math
        w_fp16 = w_int8.to(tl.float16)
        
        # Matrix Multiplication
        accumulator += tl.dot(x, w_fp16)
        
        # Advance pointers
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # 10. Multiply the result by the router's probability weight
    accumulator = accumulator * router_weights[:, None]

    # 11. Write the output back to the correct original token position
    out_ptrs = Out_ptr + (original_token_indices[:, None] * stride_outm + offs_bn[None, :] * stride_outn)
    
    # Using atomic_add because if Top-K > 1, another expert might be writing to the same token
    tl.atomic_add(out_ptrs, accumulator, mask=mask_m[:, None] & (offs_bn[None, :] < N))


def fused_bitmoe_forward(x: torch.Tensor, w: torch.Tensor, router_indices: torch.Tensor, routing_weights: torch.Tensor, num_experts: int):
    """Python wrapper to prepare memory and launch the Triton kernel."""
    M, K = x.shape
    E, K_w, N = w.shape
    top_k = router_indices.shape[1]
    
    # 1. Permutation: Sort tokens by the experts they are assigned to
    flattened_indices = router_indices.flatten()
    flattened_weights = routing_weights.flatten()
    
    # Sort the assigned experts to group tokens together
    sorted_expert_indices, sort_order = torch.sort(flattened_indices)
    
    # Find the original token indices and their corresponding weights
    sorted_token_indices = sort_order // top_k
    sorted_routing_weights = flattened_weights[sort_order]
    
    # 2. Calculate offsets so Triton knows where each expert's block of tokens begins
    expert_counts = torch.bincount(sorted_expert_indices, minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=x.device)
    torch.cumsum(expert_counts, dim=0, out=expert_offsets[1:])
    
    # 3. Allocate the output tensor
    out = torch.zeros((M, N), dtype=torch.float16, device=x.device)
    
    # 4. Define Kernel Grid and Block Sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    # Grid is 2D: (Number of token blocks per expert, Number of Experts)
    max_tokens_per_expert = expert_counts.max().item()
    grid = (triton.cdiv(max_tokens_per_expert, BLOCK_SIZE_M), num_experts)
    
    # 5. Launch the Kernel
    _fused_158b_moe_kernel[grid](
        x, w, out,
        sorted_expert_indices, sorted_routing_weights, sorted_token_indices, expert_offsets,
        M, K, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out