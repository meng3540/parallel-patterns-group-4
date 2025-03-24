#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INTENSITY_BINS 32
#define GRADIENT_BINS 32
#define EDGE_THRESHOLD 50  // Threshold to classify strong edges

// Define a 3x3 Gaussian filter kernel (normalized)
__constant__ float gaussianKernel[3][3] = {
    {1.0 / 16, 2.0 / 16, 1.0 / 16},
    {2.0 / 16, 4.0 / 16, 2.0 / 16},
    {1.0 / 16, 2.0 / 16, 1.0 / 16}
};

// CUDA Kernel for Gaussian Smoothing
__global__ void gaussianBlur(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int nx = min(max(x + j, 0), width - 1);
            int ny = min(max(y + i, 0), height - 1);
            sum += input[ny * width + nx] * gaussianKernel[i + 1][j + 1];
        }
    }
    output[y * width + x] = (unsigned char)sum;
}
__global__ void compute2DHistogram(const unsigned char* image, int width, int height,
    int* histogram, int pitch, int numBins) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int gx = 0, gy = 0;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        gx = image[(y - 1) * width + (x + 1)] - image[(y - 1) * width + (x - 1)]
            + 2 * image[y * width + (x + 1)] - 2 * image[y * width + (x - 1)]
            + image[(y + 1) * width + (x + 1)] - image[(y + 1) * width + (x - 1)];

        gy = image[(y + 1) * width + (x - 1)] - image[(y - 1) * width + (x - 1)]
            + 2 * image[(y + 1) * width + x] - 2 * image[(y - 1) * width + x]
            + image[(y + 1) * width + (x + 1)] - image[(y - 1) * width + (x + 1)];
    }

    float gradientMagnitude = sqrtf(gx * gx + gy * gy);
    int gradientBin = min(numBins - 1, (int)(gradientMagnitude / (255.0 / numBins)));
    int intensityBin = image[y * width + x] / (256 / numBins);

    atomicAdd(&histogram[intensityBin * pitch + gradientBin], 1);
}
// CUDA Kernel to compute gradients and generate an edge-detected image
__global__ void computeEdgeMap(const unsigned char* image, unsigned char* edgeMap, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int gx = 0, gy = 0;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        gx = image[(y - 1) * width + (x + 1)] - image[(y - 1) * width + (x - 1)]
            + 2 * image[y * width + (x + 1)] - 2 * image[y * width + (x - 1)]
            + image[(y + 1) * width + (x + 1)] - image[(y + 1) * width + (x - 1)];

        gy = image[(y + 1) * width + (x - 1)] - image[(y - 1) * width + (x - 1)]
            + 2 * image[(y + 1) * width + x] - 2 * image[(y - 1) * width + x]
            + image[(y + 1) * width + (x + 1)] - image[(y - 1) * width + (x + 1)];
    }

    float gradientMagnitude = sqrtf(gx * gx + gy * gy);

    // Apply threshold to highlight strong edges
    edgeMap[y * width + x] = (gradientMagnitude > EDGE_THRESHOLD) ? 255 : 0;
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

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Apply Gaussian Blur before computing gradients
    gaussianBlur << <gridSize, blockSize >> > (d_image, d_smoothed, width, height);
    cudaDeviceSynchronize();

    // Compute histogram using the smoothed image
    compute2DHistogram << <gridSize, blockSize >> > (d_smoothed, width, height, d_histogram, pitch / sizeof(int), INTENSITY_BINS);

    // Compute Edge Map
    computeEdgeMap << <gridSize, blockSize >> > (d_smoothed, d_edgeMap, width, height);
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


    // Save the edge-detected image as a raw file
    FILE* edgeFile = fopen("C:\\Users\\ruber\\Downloads\\edge_map.raw", "wb");
    fwrite(h_edgeMap, 1, imageSize, edgeFile);
    fclose(edgeFile);
    printf("Edge map saved to: C:\\Users\\ruber\\Downloads\\edge_map.raw\n");

    cudaFree(d_image);
    cudaFree(d_smoothed);
    cudaFree(d_edgeMap);
    cudaFree(d_histogram);
    free(h_histogram);
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
