#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INTENSITY_BINS 32
#define GRADIENT_BINS 32
#define EDGE_THRESHOLD 50  // Threshold for strong edges
#define TILE_SIZE 32  // Ensures 512 threads per block


// Define a 3x3 Gaussian filter kernel (normalized)
__constant__ float gaussianKernel[3][3] = {
    {1.0 / 16, 2.0 / 16, 1.0 / 16},
    {2.0 / 16, 4.0 / 16, 2.0 / 16},
    {1.0 / 16, 2.0 / 16, 1.0 / 16}
};

// CUDA Kernel for Gaussian Blur with Shared Memory
__global__ void gaussianBlur(const unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char sharedMem[TILE_SIZE + 2][TILE_SIZE + 2];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int localX = threadIdx.x + 1;
    int localY = threadIdx.y + 1;

    if (x >= width || y >= height) return;

    // Load shared memory
    sharedMem[localY][localX] = input[y * width + x];

    // Handle boundary cases
    if (threadIdx.x == 0 && x > 0)
        sharedMem[localY][localX - 1] = input[y * width + x - 1];
    if (threadIdx.x == blockDim.x - 1 && x < width - 1)
        sharedMem[localY][localX + 1] = input[y * width + x + 1];
    if (threadIdx.y == 0 && y > 0)
        sharedMem[localY - 1][localX] = input[(y - 1) * width + x];
    if (threadIdx.y == blockDim.y - 1 && y < height - 1)
        sharedMem[localY + 1][localX] = input[(y + 1) * width + x];

    __syncthreads();

    // Apply Gaussian Blur
    float sum = 0.0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            sum += sharedMem[localY + i][localX + j] * gaussianKernel[i + 1][j + 1];
        }
    }

    output[y * width + x] = (unsigned char)sum;
}

// CUDA Kernel for Edge Detection & 2D Histogram
__global__ void computeFeatures(const unsigned char* image, int width, int height,
    int* histogram, int pitch, int numBins, unsigned char* edgeMap) {
    __shared__ unsigned char sharedMem[TILE_SIZE + 2][TILE_SIZE + 2];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int localX = threadIdx.x + 1;
    int localY = threadIdx.y + 1;

    if (x >= width || y >= height) return;

    // Load shared memory
    sharedMem[localY][localX] = image[y * width + x];

    // Handle boundary cases
    if (threadIdx.x == 0 && x > 0)
        sharedMem[localY][localX - 1] = image[y * width + x - 1];
    if (threadIdx.x == blockDim.x - 1 && x < width - 1)
        sharedMem[localY][localX + 1] = image[y * width + x + 1];
    if (threadIdx.y == 0 && y > 0)
        sharedMem[localY - 1][localX] = image[(y - 1) * width + x];
    if (threadIdx.y == blockDim.y - 1 && y < height - 1)
        sharedMem[localY + 1][localX] = image[(y + 1) * width + x];

    __syncthreads();

    // Compute Gradients
    int gx = 0, gy = 0;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        gx = sharedMem[localY - 1][localX + 1] - sharedMem[localY - 1][localX - 1]
            + 2 * sharedMem[localY][localX + 1] - 2 * sharedMem[localY][localX - 1]
            + sharedMem[localY + 1][localX + 1] - sharedMem[localY + 1][localX - 1];

        gy = sharedMem[localY + 1][localX - 1] - sharedMem[localY - 1][localX - 1]
            + 2 * sharedMem[localY + 1][localX] - 2 * sharedMem[localY - 1][localX]
            + sharedMem[localY + 1][localX + 1] - sharedMem[localY - 1][localX + 1];
    }

    // Compute Gradient Magnitude
    float gradientMagnitude = sqrtf(gx * gx + gy * gy);
    int gradientBin = min(numBins - 1, (int)(gradientMagnitude / (255.0 / numBins)));
    int intensityBin = sharedMem[localY][localX] / (256 / numBins);

    // Store Edge Map
    edgeMap[y * width + x] = (gradientMagnitude > EDGE_THRESHOLD) ? 255 : 0;

    // Atomic update to histogram
    atomicAdd(&histogram[intensityBin * pitch + gradientBin], 1);
}

// Function to read a RAW grayscale image
unsigned char* readRAWImage(const char* filename, int width, int height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return NULL;
    }

    size_t imageSize = width * height;
    unsigned char* image = (unsigned char*)malloc(imageSize);

    if (fread(image, 1, imageSize, file) != imageSize) {
        printf("Error: Unexpected file size\n");
        free(image);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return image;
}

// CUDA function to compute the histogram and compare edges
void calculate2DHistogram(const unsigned char* h_image, int width, int height) {
    unsigned char* d_image, * d_smoothed, * d_edgeMap;
    int* d_histogram;

    unsigned char* h_edgeMap = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    size_t imageSize = width * height * sizeof(unsigned char);
    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&d_smoothed, imageSize);
    cudaMalloc((void**)&d_edgeMap, imageSize);
    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);

    size_t pitch;
    cudaMallocPitch((void**)&d_histogram, &pitch, GRADIENT_BINS * sizeof(int), INTENSITY_BINS);
    cudaMemset(d_histogram, 0, pitch * INTENSITY_BINS);

    dim3 blockSize(32, 16); // 512 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);



    gaussianBlur << <gridSize, blockSize >> > (d_image, d_smoothed, width, height);
    cudaDeviceSynchronize();

    computeFeatures << <gridSize, blockSize >> > (d_smoothed, width, height, d_histogram, pitch / sizeof(int), INTENSITY_BINS, d_edgeMap);
    cudaMemcpy(h_edgeMap, d_edgeMap, imageSize, cudaMemcpyDeviceToHost);
    int (*h_histogram)[GRADIENT_BINS] = (int(*)[GRADIENT_BINS])malloc(INTENSITY_BINS * GRADIENT_BINS * sizeof(int));
    cudaMemcpy2D(h_histogram, GRADIENT_BINS * sizeof(int), d_histogram, pitch, GRADIENT_BINS * sizeof(int), INTENSITY_BINS, cudaMemcpyDeviceToHost);
    // Print histogram values to terminal
    for (int i = 0; i < INTENSITY_BINS; i++) {
        for (int j = 0; j < GRADIENT_BINS; j++) {
            if (h_histogram[i][j] > 0) {
                printf("Intensity Bin: %d, Gradient Bin: %d -> Count: %d\n", i, j, h_histogram[i][j]);
            }
        }
    }

    // Save histogram to CSV file
    FILE* file = fopen("C:\\Users\\ruber\\Downloads\\histogram_output.csv", "w");
    fprintf(file, "Intensity Bin,Gradient Bin,Count\n");
    for (int i = 0; i < INTENSITY_BINS; i++) {
        for (int j = 0; j < GRADIENT_BINS; j++) {
            if (h_histogram[i][j] > 0) {
                fprintf(file, "%d,%d,%d\n", i, j, h_histogram[i][j]);
            }
        }
    }
    fclose(file);
    printf("Histogram data saved to: C:\\Users\\ruber\\Downloads\\histogram_output.csv\n");
    // Save the edge-detected image
    FILE* edgeFile = fopen("C:\\Users\\ruber\\Downloads\\edge_map.raw", "wb");
    fwrite(h_edgeMap, 1, imageSize, edgeFile);
    fclose(edgeFile);

    cudaFree(d_image);
    cudaFree(d_smoothed);
    cudaFree(d_edgeMap);
    cudaFree(d_histogram);
    free(h_edgeMap);
}

int main() {
    int width = 647;
    int height = 24;
    const char* filename = "C:\\Users\\ruber\\Downloads\\download.raw";
    unsigned char* image = readRAWImage(filename, width, height);
    if (!image) return -1;
    calculate2DHistogram(image, width, height);
    free(image);
    return 0;
}
