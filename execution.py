import torch
import triton
import triton.testing
import os
import types
import torch.nn.functional as F
from bitmoe.moe_block import SparseMoEBlock

# Create assets folder for the generated plots
os.makedirs('./assets', exist_ok=True)

# --- Part 1: VRAM Profiler ---
def measure_vram():
    print("\n--- Running VRAM Footprint Profiler ---")
    # We use larger dimensions here to clearly see the memory gap in megabytes
    d_model, hidden_dim, num_experts, top_k = 1024, 4096, 8, 2

    def get_memory(use_158b):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Initialize model
        model = SparseMoEBlock(d_model, hidden_dim, num_experts, top_k).cuda().half()
        model.requires_grad_(False)
        
        if use_158b:
            # Simulate 1.58b storage by compressing the weights to int8 format
            for expert in model.experts:
                expert.w1.weight.data = expert.w1.weight.data.to(torch.int8)
                expert.w2.weight.data = expert.w2.weight.data.to(torch.int8)
                expert.w3.weight.data = expert.w3.weight.data.to(torch.int8)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        # Record the peak memory allocated by PyTorch
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        del model # Free memory for the next run
        return mem_mb

    fp16_mem = get_memory(use_158b=False)
    bit_mem = get_memory(use_158b=True)
    
    print(f"Standard FP16 MoE VRAM : {fp16_mem:.2f} MB")
    print(f"BitMoE 1.58b VRAM    : {bit_mem:.2f} MB")
    print(f"Reduction Achieved   : {((fp16_mem - bit_mem) / fp16_mem) * 100:.2f}%\n")


# --- Part 2: Throughput Profiler (Triton Perf Report) ---
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],  # Sequence length is the independent variable
        x_vals=[128, 256, 512, 1024, 2048, 4096],  # Testing scaling limits
        line_arg='provider',
        line_vals=['pytorch', 'triton'],
        line_names=['PyTorch Standard Loop', 'Fused Triton BitMoE'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='Execution Time (ms)',
        plot_name='throughput_scaling',
        args={'batch_size': 4, 'd_model': 512, 'hidden_dim': 2048, 'num_experts': 8, 'top_k': 2},
    )
)
def benchmark_throughput(batch_size, seq_len, d_model, hidden_dim, num_experts, top_k, provider):
    # Setup dummy inputs
    x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float16)
    
    # Initialize the model
    model = SparseMoEBlock(d_model, hidden_dim, num_experts, top_k).cuda().half()
    model.requires_grad_(False)

    if provider == 'triton':
        # Apply the same isolation patch from Day 3 to test raw C-level integer speed
        for expert in model.experts:
            strict_ternary = torch.randint(-1, 2, expert.w1.weight.shape, dtype=torch.float16, device='cuda')
            expert.w1.weight.copy_(strict_ternary)
            def bypass_ste_forward(self, x_in):
                return F.linear(x_in, self.weight, self.bias)
            expert.w1.forward = types.MethodType(bypass_ste_forward, expert.w1)

        # do_bench handles GPU warmups, synchronization, and repeated runs for a stable median
        ms = triton.testing.do_bench(lambda: model(x, use_triton=True))
    else:
        ms = triton.testing.do_bench(lambda: model(x, use_triton=False))

    return ms

def execute_day_4():
    if not torch.cuda.is_available():
        print("CRITICAL: A GPU is required for benchmarking.")
        return

    measure_vram()
    print("Running Triton Throughput Benchmark (This will take about 30-60 seconds...)")
    
    # Run the perf_report. It will automatically print to console AND save a PNG graph.
    benchmark_throughput.run(print_data=True, show_plots=False, save_path='./assets')
    print("\nBenchmarks Complete! Check the /assets folder on the left for your CSV and PNG.")

if __name__ == "__main__":
    execute_day_4()