import torch
import matplotlib.pyplot as plt
import numpy as np
from bitmoe.moe_block import SparseMoEBlock
from bitmoe.experts.bitlinear import TernaryQuantizeSTE

def execute_day_2():
    torch.manual_seed(42)
    d_model, hidden_dim, num_experts, top_k = 128, 512, 8, 2
    batch_size, seq_len = 2, 512
    
    model = SparseMoEBlock(d_model, hidden_dim, num_experts, top_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    target = torch.randn(batch_size, seq_len, d_model)

    # Tracking metrics for the Visuals
    steps = 500
    temperatures = []
    gradient_magnitudes = []

    print("Initiating Soft-Start Training Loop...")
    for step in range(steps):
        # Temperature Annealing Logic: 5.0 decaying to 1.0
        current_temp = max(1.0, 5.0 * (1 - step / (steps * 0.5)))
        temperatures.append(current_temp)
        
        optimizer.zero_grad()
        output, l_bal = model(dummy_input, temperature=current_temp)
        
        task_loss = torch.nn.functional.mse_loss(output, target)
        total_loss = task_loss + (0.1 * l_bal)
        total_loss.backward()
        
        # Track gradient stability of Expert 0's first layer
        grad_mag = model.experts[0].w1.weight.grad.abs().mean().item()
        gradient_magnitudes.append(grad_mag)
        
        optimizer.step()

    # --- Generate Visual 2: Weight Distribution Histogram ---
    print("Generating Visual 2: Weight Distribution...")
    raw_weights = model.experts[0].w1.weight.detach().flatten()
    quantized_weights = TernaryQuantizeSTE.apply(model.experts[0].w1.weight).detach().flatten()
    
    plt.figure(figsize=(10, 5))
    plt.hist(raw_weights.numpy(), bins=50, alpha=0.5, label='Latent FP32 Weights', color='lightblue')
    plt.hist(quantized_weights.numpy(), bins=50, alpha=0.8, label='Quantized 1.58-bit Weights {-1, 0, 1}', color='darkblue')
    plt.title("BitLinear Weight Distribution (The Dirac Delta)")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("weight_distribution.png")
    plt.close()

    # --- Generate Visual 3: Soft-Start Annealing Curve ---
    print("Generating Visual 3: Soft-Start Annealing...")
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Softmax Temperature (T)', color=color)
    ax1.plot(temperatures, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Expert Gradient Magnitude', color=color)  
    ax2.plot(gradient_magnitudes, color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Soft-Start Heuristic: Temperature Annealing vs Gradient Stability")
    fig.tight_layout()
    plt.savefig("soft_start_annealing.png")
    plt.close()

    print("Day 2 execution complete. Check assets for Visual 2 and 3.")

if __name__ == "__main__":
    execute_day_2()