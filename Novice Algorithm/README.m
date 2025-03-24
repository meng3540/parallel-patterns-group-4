Code Description

This CUDA C program calculates a 2D histogram of an image based on pixel intensity and gradient magnitude, and saves the result to a CSV file. The program uses parallel computing via CUDA to speed up the calculation, and processes the image using the Sobel operator to compute gradients.

Header Files and Macros: cuda_runtime.h:

Provides functions for memory management, kernel launches, and handling device operations in CUDA. stdio.h: Standard I/O functions for file operations (e.g., writing the histogram to a CSV). stdlib.h: Provides memory allocation functions. math.h: Provides mathematical functions like sqrt for calculating gradient magnitude. Macros:

INTENSITY_BINS (256): Number of bins for intensity values (0-255 range). GRADIENT_BINS (256): Number of bins for gradient magnitudes (0-255 range).

Kernel: compute2DHistogram This CUDA kernel is responsible for calculating the gradient magnitude and populating the 2D histogram:

Inputs:

image: Pointer to the image data (grayscale values). width, height: Dimensions of the image. histogram: The 2D histogram array (intensity vs. gradient). pitch: The width of the histogram in memory to handle memory alignment. numBins: The number of bins for intensity and gradient. Process:

The kernel assigns a thread for each pixel in the image using its position (blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y). The Sobel operator is applied to compute the horizontal and vertical gradients (gx, gy). Intensity Bin: The intensity of the pixel is directly mapped into a bin (0-255).

Atomic Operation:

The atomicAdd function is used to update the histogram, ensuring that multiple threads writing to the same memory location do not cause race conditions.

Host Function: writeHistogramToCSV This function writes the computed 2D histogram to a CSV file:

CSV Format: Each row in the CSV represents a non-zero entry in the histogram. Columns include the Intensity Bin, Gradient Bin, and Count (the frequency of pixels in that bin combination).

Process: It loops through the histogram array, checking for non-zero values. Each non-zero entry is written to the CSV file in the format: Intensity Bin, Gradient Bin, Count.

Host Function: calculate2DHistogram This function manages memory allocation, kernel execution, and saving the histogram:

Memory Allocation: Allocates device memory for the grayscale image (d_image). Allocates device memory for the histogram using cudaMallocPitch to handle 2D memory layout, ensuring proper alignment. Data Transfer: Copies the image from host memory to device memory using cudaMemcpy. Initializes the histogram on the device using cudaMemset to set all bins to zero. Kernel Launch: Defines the block size (16x16 threads) and grid size (calculated based on image dimensions). Launches the compute2DHistogram kernel to process the image in parallel. Data Transfer Back: After kernel execution, copies the 2D histogram from device memory back to host memory using cudaMemcpy2D. Write Histogram: Passes the histogram data to the writeHistogramToCSV function to save it as a CSV file. Memory Cleanup: Frees both host and device memory after processing.

Host Function: readImage This function reads a grayscale image:

Dummy Data: For testing purposes, the image is filled with random intensity values between 0 and 255. This simulates loading a real grayscale image. Image Dimensions: It assigns fixed dimensions to the image (512x512 pixels in this case). Real-world Replacement: In a real scenario, this function would read the image from a file (e.g., using libraries like stb_image for PNG/JPEG or other formats).

Main Function: main This is the entry point of the program:

Image Reading: Calls the readImage function to load the image data (or generate random values for testing). Histogram Calculation: The calculate2DHistogram function is called to perform the image processing and histogram computation on the GPU. Free Memory: After the histogram is written to the CSV file, the allocated memory (both host and device) is freed. Key Concepts and Techniques: Parallel Computing with CUDA:

The program utilizes GPU parallelism to process each pixel independently, leveraging CUDA’s thread-level parallelism for faster computation. Thread Indexing: Each thread handles a specific pixel in the image, with blockIdx and threadIdx used to calculate its unique position. Sobel Gradient Calculation:

The Sobel operator is used to compute the gradient of the image in both the x and y directions, helping to detect edges and changes in intensity. 2D Histogram:

The histogram represents the relationship between pixel intensity and gradient magnitude. Histogram Binning: Both the intensity and gradient magnitudes are divided into bins to categorize pixels based on these values. Atomic Operations:

The use of atomicAdd ensures that when multiple threads try to update the same histogram bin, they do so safely without causing race conditions.

CSV Output:

The histogram is saved in a CSV file, making it easy to analyze and visualize the results outside the program.
