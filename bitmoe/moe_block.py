import torch
import torch.nn as nn
import torch.nn.functional as F
from .routing.router import TopKRouter
from .experts.expert_layer import BitExpert
from .routing.loss import calculate_load_balancing_loss
from .kernels.fused_dispatch import fused_bitmoe_forward

class SparseMoEBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            [BitExpert(d_model, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0, use_triton: bool = False):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        scaled_logits = self.router(x_flat) / temperature
        l_bal = calculate_load_balancing_loss(scaled_logits, self.num_experts, self.top_k)
        
        routing_weights, selected_experts = torch.topk(scaled_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        if use_triton and x.is_cuda:
            # Stack the quantized weights from all experts into a single contiguous tensor
            # Shape: (num_experts, d_model, hidden_dim)
            stacked_w1 = torch.stack([expert.w1.weight.t() for expert in self.experts]).contiguous()
            
            # Note: For brevity in this sprint, we are only running the first layer of the FFN (w1) 
            # through the Triton kernel to prove the concept. A full production block would chain w1, silu, w3, w2.
            final_output = fused_bitmoe_forward(
                x_flat, stacked_w1, selected_experts, routing_weights, self.num_experts
            )
            return final_output.view(batch_size, seq_len, self.hidden_dim), l_bal

        else:
            # Day 2 PyTorch Fallback
            final_output = torch.zeros(x_flat.size(0), self.hidden_dim, dtype=x.dtype, device=x.device)
            for i, expert in enumerate(self.experts):
                expert_mask = (selected_experts == i).any(dim=-1)
                if expert_mask.any():
                    expert_tokens = x_flat[expert_mask]
                    # Executing just w1 to match the Triton test scope
                    expert_output = expert.w1(expert_tokens)
                    weight_mask = (selected_experts[expert_mask] == i)
                    token_weights = routing_weights[expert_mask][weight_mask].unsqueeze(-1)
                    final_output[expert_mask] += expert_output * token_weights
            return final_output.view(batch_size, seq_len, self.hidden_dim), l_bal