import torch
import matplotlib.pyplot as plt
from bitmoe.moe_block import SparseMoEBlock

def simulate_routing(use_loss_penalty=False):
    torch.manual_seed(42) # For reproducible "bad" initialization
    d_model, hidden_dim, num_experts, top_k = 128, 512, 8, 2
    batch_size, seq_len = 2, 512 # 1024 total tokens
    
    model = SparseMoEBlock(d_model, hidden_dim, num_experts, top_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # We create an artificial target just to make the network train
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    target = torch.randn(batch_size, seq_len, d_model)

    expert_counts = [0] * num_experts

    for step in range(50):
        optimizer.zero_grad()
        output, l_bal = model(dummy_input)
        
        # Main task loss + conditional auxiliary loss
        task_loss = torch.nn.functional.mse_loss(output, target)
        total_loss = task_loss + (0.1 * l_bal if use_loss_penalty else 0.0)
        
        total_loss.backward()
        optimizer.step()

    # Capture final routing distribution
    with torch.no_grad():
        gate_logits = model.router(dummy_input.view(-1, d_model))
        _, selected_experts = torch.topk(gate_logits, top_k, dim=-1)
        for idx in selected_experts.flatten().tolist():
            expert_counts[idx] += 1
            
    return expert_counts

# Run both simulations
print("Simulating Router Collapse (No Penalty)...")
counts_collapsed = simulate_routing(use_loss_penalty=False)

print("Simulating Balanced Routing (With L_bal Penalty)...")
counts_balanced = simulate_routing(use_loss_penalty=True)

# Generate the Visual for the README
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

experts = [f"E{i+1}" for i in range(8)]
ax1.bar(experts, counts_collapsed, color='salmon')
ax1.set_title("Router Collapse (L_bal = 0.0)")
ax1.set_ylabel("Tokens Routed")

ax2.bar(experts, counts_balanced, color='lightgreen')
ax2.set_title("Optimal Load Balancing (L_bal = 0.1)")

plt.suptitle("Impact of Auxiliary Loss on Expert Utilization", fontsize=14)
plt.tight_layout()
plt.savefig("router_collapse.png")
print("Saved visual to router_collapse.png! Day 1 Complete.")