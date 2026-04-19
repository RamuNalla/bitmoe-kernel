import torch
import torch.nn as nn

class TopKRouter(nn.Module):
    """Generates routing probabilities and Top-K indices."""
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size * seq_len, d_model)
        gate_logits = self.gate(x)
        return gate_logits
        