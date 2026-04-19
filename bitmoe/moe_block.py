import torch
import torch.nn as nn
import torch.nn.functional as F
from .routing.router import TopKRouter
from .experts.expert_layer import StandardExpert
from .routing.loss import calculate_load_balancing_loss

class SparseMoEBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            [StandardExpert(d_model, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # Flatten for routing
        
        # 1. Get routing logits
        gate_logits = self.router(x_flat)
        
        # 2. Calculate Load Balancing Loss
        l_bal = calculate_load_balancing_loss(gate_logits, self.num_experts, self.top_k)
        
        # 3. Extract Top-K weights and indices
        routing_weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # 4. Standard PyTorch Dispatch (Memory Heavy, Slow)
        final_output = torch.zeros_like(x_flat)
        
        # Iterate through experts (This loop is what we will kill with Triton on Day 3)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == i).any(dim=-1)
            
            if expert_mask.any():
                expert_tokens = x_flat[expert_mask]
                
                # Forward pass through the specific expert
                expert_output = expert(expert_tokens)
                
                # Fetch the routing weight for this token-expert pair
                weight_mask = (selected_experts[expert_mask] == i)
                token_weights = routing_weights[expert_mask][weight_mask].unsqueeze(-1)
                
                # Add the weighted output to the final tensor
                final_output[expert_mask] += expert_output * token_weights
                
        return final_output.view(batch_size, seq_len, d_model), l_bal