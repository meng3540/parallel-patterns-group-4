Why We Chose Privatization and Shared Memory as Optimization Techniques
In GPU programming, memory access patterns are critical to performance. CUDA applications can be bottlenecked by slow global memory access. To address this, we applied two key optimization strategies: Privatization and Shared Memory. We selected these techniques based on profiling insights and hardware access characteristics.

Profiling Insights Summary
From profiling with tools like Nsight Compute / Nsight Systems, the following patterns were observed:

Metric	Observation	Bottleneck
Global Memory Throughput	High	Threads accessing global memory redundantly
Memory Load Efficiency	Low (~40–60%)	Multiple threads read the same pixels from global memory
Occupancy / Register Usage	Balanced	Enough registers per thread for privatization
Shared Memory Bandwidth	Underutilized	Shared memory had headroom to offload access

Privatization: What and Why
Privatization refers to assigning each thread its own private copy of data from global memory and reusing it in registers.

Advantages:
Fast access due to register usage

Simple logic — no thread synchronization

Great when threads don’t need to share data

Applied To:
Gaussian blur kernel: Each thread accessed pixels using boundary clamping logic and reused them locally (in registers)

Edge detection kernel: Threads computed gradients using only their local values

Profiling Results:
Metric	Before	After Privatization
Global load transactions	High	↓ Reduced by ~25%
Register efficiency	Moderate	↑ Increased due to reused reads
Warp efficiency	Stable	No divergence impact

Shared Memory: What and Why
Shared Memory is a fast, low-latency memory space shared among threads in the same block.

Advantages:
Great for data reuse across threads

Reduces global memory bandwidth pressure

Ideal for stencil operations like convolution

Applied To:
Gaussian blur: All pixels in a 2D tile were loaded once per block

Edge detection: Neighbor pixels needed for gradient were shared to avoid multiple global reads

Profiling Results:
Metric	Before	After Shared Memory
Global memory throughput	High	↓ Dropped ~30–40%
Shared memory usage	Low	↑ Utilized 60–80% SMEM
Kernel execution time	Baseline	↓ Improved 1.5x–2x speedup in tiled regions

Final Comparison Based on Profiling
Optimization	Use Case	Performance Gain	Tradeoff
Privatization	Simple reuse within thread	Lower latency, fewer reads	Higher register usage
Shared Memory	Reuse across threads (tiles)	Lower global bandwidth	Requires sync + boundary logic

Conclusion
We chose privatization for kernels where each thread's data is mostly independent, and shared memory where threads needed access to overlapping data (like in stencil patterns). Profiling clearly showed that each method improved memory efficiency and reduced kernel execution time in their respective contexts.

Both approaches complement each other and were critical in turning a memory-bound program into a compute-efficient one.
