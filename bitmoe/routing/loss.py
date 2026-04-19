import torch
import torch.nn.functional as F

def calculate_load_balancing_loss(
    gate_logits: torch.Tensor, 
    num_experts: int, 
    top_k: int
) -> torch.Tensor:
    """
    Calculates the auxiliary load balancing loss to prevent router collapse.
    """
    # 1. Soft probabilities across all experts
    routing_probs = F.softmax(gate_logits, dim=-1)
    
    # 2. Hard routing decisions (which experts were actually chosen)
    _, selected_experts = torch.topk(gate_logits, top_k, dim=-1)
    
    # Create a boolean mask of shape (batch * seq_len, num_experts)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).float()
    
    # 3. f_i: Fraction of tokens routed to each expert
    tokens_per_expert = expert_mask.sum(dim=(0, 1)) # Sum across batch and top_k
    total_tokens = gate_logits.size(0) * top_k
    f_i = tokens_per_expert / total_tokens
    
    # 4. P_i: Mean routing probability for each expert
    P_i = routing_probs.mean(dim=0)
    
    # 5. The Loss formula: N * sum(f_i * P_i)
    loss = num_experts * torch.sum(f_i * P_i)
    return loss