#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INTENSITY_BINS 32
#define GRADIENT_BINS 32
#define EDGE_THRESHOLD 50
#define TILE_SIZE 32

// 3x3 Gaussian Kernel in Constant Memory
__constant__ float gaussianKernel[3][3] = {
    {1.0f / 16, 2.0f / 16, 1.0f / 16},
    {2.0f / 16, 4.0f / 16, 2.0f / 16},
    {1.0f / 16, 2.0f / 16, 1.0f / 16}
};

// Gaussian Blur Kernel (Privatization Only)
__global__ void gaussianBlur(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int nx = fminf(fmaxf(x + j, 0), width - 1);
            int ny = fminf(fmaxf(y + i, 0), height - 1);
            sum += input[ny * width + nx] * gaussianKernel[i + 1][j + 1];
        }
    }

    output[y * width + x] = (unsigned char)sum;
}

// Edge Detection & 2D Histogram (Privatization Only)
__global__ void computeFeatures(const unsigned char* image, int width, int height,
    int* histogram, int pitch, int numBins, unsigned char* edgeMap) {

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

    float gradMag = sqrtf((float)(gx * gx + gy * gy));
    int gradientBin = min(numBins - 1, (int)(gradMag / (255.0f / numBins)));
    int intensityBin = image[y * width + x] / (256 / numBins);

    edgeMap[y * width + x] = (gradMag > EDGE_THRESHOLD) ? 255 : 0;

    atomicAdd(&histogram[intensityBin * pitch + gradientBin], 1);
}

// Read RAW Grayscale Image
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

// Host-side Execution and Output
void calculate2DHistogram(const unsigned char* h_image, int width, int height) {
    unsigned char* d_image, * d_smoothed, * d_edgeMap;
    int* d_histogram;
    unsigned char* h_edgeMap = (unsigned char*)malloc(width * height);

    size_t imageSize = width * height * sizeof(unsigned char);
    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_smoothed, imageSize);
    cudaMalloc(&d_edgeMap, imageSize);
    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);

    size_t pitch;
    cudaMallocPitch((void**)&d_histogram, &pitch, GRADIENT_BINS * sizeof(int), INTENSITY_BINS);
    cudaMemset(d_histogram, 0, pitch * INTENSITY_BINS);

    dim3 blockSize(32, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    gaussianBlur << <gridSize, blockSize >> > (d_image, d_smoothed, width, height);
    cudaDeviceSynchronize();

    computeFeatures << <gridSize, blockSize >> > (d_smoothed, width, height, d_histogram,
        pitch / sizeof(int), INTENSITY_BINS, d_edgeMap);
    cudaDeviceSynchronize();

    cudaMemcpy(h_edgeMap, d_edgeMap, imageSize, cudaMemcpyDeviceToHost);

    int (*h_histogram)[GRADIENT_BINS] = (int(*)[GRADIENT_BINS])malloc(INTENSITY_BINS * GRADIENT_BINS * sizeof(int));
    cudaMemcpy2D(h_histogram, GRADIENT_BINS * sizeof(int),
        d_histogram, pitch, GRADIENT_BINS * sizeof(int), INTENSITY_BINS, cudaMemcpyDeviceToHost);

    // ✅ Declare file BEFORE using
    FILE* file = fopen("C:\\Users\\n01595458\\Downloads\\histogram_output.csv", "w");
    if (file) {
        fprintf(file, "Intensity Bin,Gradient Bin,Count\n");
        for (int i = 0; i < INTENSITY_BINS; i++) {
            for (int j = 0; j < GRADIENT_BINS; j++) {
                if (h_histogram[i][j] > 0) {
                    fprintf(file, "%d,%d,%d\n", i, j, h_histogram[i][j]);
                }
            }
        }
        fclose(file);
    }

    FILE* edgeFile = fopen("C:\\Users\\n01595458\\Downloads\\edge_map.raw", "wb");
    if (edgeFile) {
        fwrite(h_edgeMap, 1, imageSize, edgeFile);
        fclose(edgeFile);
    }

    cudaFree(d_image);
    cudaFree(d_smoothed);
    cudaFree(d_edgeMap);
    cudaFree(d_histogram);
    free(h_edgeMap);
    free(h_histogram);
}

// Main Entry Point
int main() {
    int width = 647;
    int height = 24;
    const char* filename = "C:\\Users\\n01595458\\Downloads\\download.raw";
    unsigned char* image = readRAWImage(filename, width, height);
    if (!image) return -1;

    calculate2DHistogram(image, width, height);
    free(image);
    return 0;
}
