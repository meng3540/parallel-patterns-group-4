
![Captura de pantalla 2025-03-25 163558](https://github.com/user-attachments/assets/817af5f0-0595-41ba-ba94-0d0831e6be67)


1. Memory Throughput
Represents how much data is transferred between memory and GPU cores.

Higher = more data movement, but not always efficient.
No optimized: 17.07% → high, but potentially due to inefficient global memory use.
Optimized: 6.93% → improved memory access pattern, less waste.
Only shared: 3.04% → fewer memory accesses, but maybe not fully optimized.
Only Privatization: 5.78% → decent balance.
Optimized code reduces unnecessary memory traffic, making it more efficient.

3. Running Time
Execution time of the kernel.
Lower = better (faster).
No optimized: 67.1 microseconds.
Optimized: 21.76 μs → ~3× faster.
Only shared: 36.76 μs.
Only Privatization: 19.26 μs → fastest.
Privatization alone performs best in speed, but full optimization gives both speed and balance.

4. SM Throughput
How effectively the GPU Streaming Multiprocessors are used.
Higher = more work done per cycle.
No optimized: 29.39%.
Optimized: 32.49% → best.
Only shared: 18.97%.
Only Privatization: 18.52%.
Full optimization maximizes SM usage. Privatization improves speed, but doesn't fully utilize GPU cores.

5. Occupancy
How many threads are running relative to hardware limits.
Higher = better resource utilization.
No optimized: 61.79%.
Optimized: 78.55% → best.
Only shared: 27.09%.
Only Privatization: 26.17%.
Optimized code launches more threads efficiently.

5. Total SM Elapsed Cycles
Total GPU cycles used by the kernel.
Lower = more efficient.
No optimized: 928,472 cycles.
Optimized: 995,462 cycles.
Only shared: 1,614,532 cycles → worst.
Only Privatization: 835,346 → lowest.
Despite more cycles, the optimized version balances occupancy, throughput, and memory use. Privatization wins in raw speed but not in SM throughput or occupancy.

Conclusion:

Metric	Best Performer
Running Time	Only Privatization
Memory Efficiency	Optimized
SM Throughput	Optimized
Occupancy	Optimized
Least Cycles	Only Privatization

Optimized method provides the best overall performance across all metrics.


