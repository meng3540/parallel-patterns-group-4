Understanding the Motivation Behind the Optimizations

When we initially profiled the CUDA image processing code, we noticed that a significant amount of time was being spent on global memory accesses. The kernels involved repetitive fetching of neighboring pixel values for convolution and edge detection operations. This led to inefficient memory access patterns, high global memory bandwidth consumption, and redundant reads across multiple threads. These observations guided our decision to apply two well-known optimization techniques: privatization and shared memory.

Why We Used Privatization

Privatization is an optimization strategy where each thread stores the data it needs into its own private registers. In the context of our Gaussian blur and edge detection kernels, privatization allowed each thread to compute its result using values it directly fetched from global memory once and reused in local registers. This eliminated redundant accesses for each thread’s own data. Since each thread operated independently and didn’t need to coordinate with others, the logic remained simple and fast.

From the profiling perspective, once privatization was introduced, we observed a decrease in global memory transaction count, especially for data that didn't need to be shared. Additionally, register utilization increased slightly, which is expected with this technique, but remained within the limits of occupancy. The key gain from privatization was faster access to frequently reused data and reduced pressure on global memory bandwidth. This translated into measurable speedups during kernel execution in regions of the code where each thread’s data footprint was isolated.

Why We Used Shared Memory

In contrast, shared memory was chosen for parts of the code where multiple threads accessed overlapping regions of the input image — a typical pattern in convolution and edge detection. For instance, in a 3×3 stencil used in Gaussian blur, neighboring threads all needed access to some of the same pixel values. Accessing those shared pixels from global memory repeatedly across threads was highly inefficient.

By introducing shared memory, we made it possible to load blocks (or tiles) of the image into fast, low-latency memory shared by all threads in the block. Threads then accessed the shared tile data instead of reading directly from global memory. This dramatically reduced the number of global memory reads and allowed threads to compute their output more efficiently. The introduction of shared memory did require synchronization between threads (__syncthreads()) and careful handling of boundary conditions (halo regions), but the performance benefits outweighed the added complexity.

After applying shared memory, the profiling results showed a noticeable reduction in global memory throughput and a substantial increase in shared memory utilization. Kernel execution times decreased, particularly for the Gaussian blur stage, which benefited heavily from data reuse. The improved locality of reference and reduced latency from shared memory access helped the kernel sustain higher throughput and better cache behavior.

Final Reflection

The decision to apply privatization and shared memory was directly influenced by profiling data. Where threads worked independently on their own pixels, privatization helped by cutting down unnecessary repeated global reads. Where threads needed access to common data (especially in stencil-based computations), shared memory provided a clear advantage by allowing tile-based reuse and significantly reducing bandwidth usage.

Both techniques target the same problem — global memory inefficiency — but from different angles. Privatization is lightweight and thread-local, best for isolated data. Shared memory is powerful for collaborative access within thread blocks, ideal for structured patterns. Together, they allowed us to tailor the memory access behavior to the needs of each kernel, achieving a more efficient and scalable CUDA implementation.
