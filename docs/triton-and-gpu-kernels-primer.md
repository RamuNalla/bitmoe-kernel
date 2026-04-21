# GPU kernels, Triton, and CUDA — primer

This document collects a systematic introduction to **GPU kernels**, **Triton**, and **CUDA**, including a minimal Triton example and notes on why Triton is useful for developers.

---

## Table of contents

1. [What is a kernel? (from basics)](#1-what-is-a-kernel-from-basics)
2. [What is Triton?](#2-what-is-triton)
3. [What is CUDA?](#3-what-is-cuda)
4. [Triton vs CUDA](#4-triton-vs-cuda)
5. [Minimal Triton example: vector add](#5-minimal-triton-example-vector-add)
6. [Why Triton helps developers](#6-why-triton-helps-developers)
7. [Naming note: “Triton” collisions](#7-naming-note-triton-collisions)

---

## 1. What is a kernel? (from basics)

The word **kernel** is overloaded. In **GPU and high-performance computing**, a **kernel** usually means:

> A single function that you launch on the device (often the GPU) so that it runs in parallel many times—once per thread (or per “work item”)—over some large dataset.

It is not the whole application. It is the **small, hot inner routine** executed **at massive scale** in parallel.

### CPU vs GPU mental model

On a **CPU**, you might write a loop:

```text
for i in range(N):
    y[i] = f(x[i])
```

On a **GPU**, you often write a **kernel** that specifies what **one** index (or tile) should do; the runtime **schedules many workers** so all indices are covered concurrently.

### Typical execution flow

1. **Host** (CPU): prepares data in GPU memory and **launches** the kernel.
2. **Device** (GPU): runs **many lightweight threads** (grouped into warps/blocks on NVIDIA hardware).
3. Each worker determines **which element(s)** it owns (via indices such as `threadIdx` / `blockIdx` in CUDA, or Triton’s program/block model).
4. All workers run the **same program** on **different data** (**SPMD**: single program, multiple data).

So the kernel is the **“single program”** in SPMD.

### Why the name “kernel”

It is the **core** of the computation—the inner loop lifted out and run in parallel—rather than surrounding orchestration (allocation, I/O, the Python interpreter, etc.).

### Not the operating system kernel

In OS terminology, the **kernel** is the core of the operating system. In **GPU programming**, **kernel** almost always means a **launchable parallel function** on the GPU, not the OS.

### Connection to Triton / CUDA

Both Triton and CUDA are ways to **author** that parallel function and **compile** or **lower** it to efficient GPU execution.

---

## 2. What is Triton?

**Triton** (commonly **OpenAI Triton** or the **Triton language**) is a **Python-based language and compiler** for writing **custom GPU kernels**.

### Problem it addresses

Peak performance on NVIDIA GPUs has traditionally required **CUDA C++** or heavy use of vendor libraries. That is powerful but involves a lot of low-level detail: threads, shared memory, and tuning.

Triton lets you express kernels in **Python-like code** (e.g. functions decorated with `@triton.jit`). The **Triton compiler** turns that into **efficient GPU code**.

### Typical use

- **PyTorch**: Custom or fused operations; parts of **Torch Inductor** / `torch.compile` may generate or call Triton kernels.
- **Research** (e.g. MoE, attention): Fusing dispatch, softmax, matmuls, etc. into one kernel to reduce memory traffic and launch overhead.

### One-line summary

**Triton is a Python DSL plus compiler for writing and tuning GPU kernels**, often with less boilerplate than hand-written CUDA for many ML-style workloads.

---

## 3. What is CUDA?

**CUDA** is **NVIDIA’s platform** for running general-purpose computation on **NVIDIA GPUs**.

### What you write

- Code in **C/C++** with CUDA extensions (or APIs that compile to the same execution model).
- A **GPU kernel** is a function that runs on the GPU, launched from the **CPU (host)**.
- Launches specify **how much parallelism**: a **grid of thread blocks**, each with many **threads**.

### Mental model

- **Host (CPU)**: memory setup, data movement, kernel launches, coordination.
- **Device (GPU)**: executes the kernel **many times in parallel**; each thread runs the **same instructions** on **different data** (SPMD).

### What CUDA provides

- **Direct control** over thread indices, **shared memory**, synchronization within a block, and performance-oriented choices.
- A mature **toolchain** (`nvcc`) and libraries (e.g. cuBLAS, cuDNN).

### One-line summary

**CUDA is NVIDIA’s primary way to program NVIDIA GPUs**, usually with explicit grids of thread blocks and C/C++-style kernels.

---

## 4. Triton vs CUDA

| | **CUDA** | **Triton** |
|---|----------|------------|
| **Typical language** | C/C++ with CUDA | Python (Triton DSL) |
| **Control** | Very explicit (threads, blocks, memory) | Higher level; compiler handles more of the mapping |
| **Ecosystem** | Native NVIDIA stack | Very common on NVIDIA GPUs in ML; “kernel compiler” mindset |
| **Strength** | Maximum control, deep integration with NVIDIA libraries | Fast iteration for ML kernels, fusion, custom layouts |

**Shared idea:** both define a **GPU kernel**—a parallel function launched from the host that runs at large scale on the GPU. **CUDA** is the foundational NVIDIA model; **Triton** is a Python-centric way to author kernels that compile to strong GPU implementations.

---

## 5. Minimal Triton example: vector add

Illustrative **vector add** on GPU tensors: shows programs, blocking, masks, and launch grid.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Launched many times in parallel ("programs").
    # Each program handles a contiguous chunk of size BLOCK_SIZE.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Last block may be partial if n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    out = torch.empty_like(x)
    n = x.numel()

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out
```

### How this works

1. **`add_kernel` is the GPU kernel** — the body uses **Triton’s DSL** (`tl.load`, `tl.store`, `tl.arange`, …), not ordinary sequential Python on the CPU.
2. **`@triton.jit`** — marks the function for **JIT compilation** to GPU code.
3. **`tl.program_id`** — each parallel instance gets `pid`, so different programs handle **different chunks** of the 1-D tensor.
4. **`BLOCK_SIZE` and `tl.arange`** — each program processes **BLOCK_SIZE** elements per iteration (here 1024), a common blocking pattern.
5. **`mask`** — bounds-safe loads/stores when the last block is partial.
6. **`add_kernel[grid](...)`** — **launch**: run the kernel over `grid` programs; `triton.cdiv(n, BLOCK_SIZE)` is the number of programs needed along axis 0.
7. **PyTorch tensors** — passed as **raw pointers** to GPU memory (`x_ptr`, etc.).

For the **simplest** operations, well-tuned libraries may already match or beat a naive kernel. The pattern matters for understanding launches and blocking before moving to matmul tiles and fusion.

---

## 6. Why Triton helps developers

### 1. One ecosystem (often Python + PyTorch)

Custom CUDA often implies **C++/CUDA sources**, **build systems**, and cross-language debugging. Triton keeps kernel logic **next to** training/experiment code for faster iteration.

### 2. Less boilerplate for parallel array logic

CUDA exposes threads, blocks, and warps explicitly. Triton’s **program + tile** view often aligns with how people think about **chunks of arrays**, especially for elementwise and blocked numeric kernels.

### 3. Fusion (main practical win)

Multiple steps can be combined in **one kernel**:

- read inputs once,
- compute several ops in the same kernel,
- write once,

reducing **memory bandwidth** and **kernel launch** overhead. This is a major reason Triton is used for **custom layers**, **attention**, and **MoE-style** code paths.

### 4. Compiler and tuning

The author writes a structured description; the toolchain optimizes lowering. Real projects still tune block sizes, warps, etc. (sometimes with autotuning), but avoid maintaining raw CUDA for every small variant.

### Caveat

For **trivial** ops already covered by optimized libraries, Triton is not automatically faster. Value is highest for **custom access patterns**, **layouts**, and **fused pipelines** that are not a single library call.

---

## 7. Naming note: “Triton” collisions

- In **ML / GPU code**, **Triton** usually means **OpenAI’s Triton language and compiler** for kernels.
- **NVIDIA Triton Inference Server** is a **different** product (model serving). Do not confuse the two when reading docs or job posts.

---

## Related reading in this repo

- Project README: [`README.md`](../README.md)
- Example kernel code may live under `bitmoe/kernels/` (e.g. fused dispatch and other Triton entry points).
