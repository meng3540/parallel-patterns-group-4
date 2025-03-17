// Include libraries for CUDA programming, and math operations
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Defining the number of bins for intensity and gradient values
#define INTENSITY_BINS 256  
#define GRADIENT_BINS 256   

// CUDA Kernel to populate the 2D histogram
// Each thread calculates one pixel's gradient and updates the histogram 
_global_ void compute2DHistogram(const unsigned char* image, int width, int height,
                                 int* histogram, int pitch, int numBins) {
    // Calculate the thread's coordinates in the image grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Prevent to go out of image bondary 
    if (x >= width || y >= height) return;

    // Defining variables to store horizontal (gx) and vertical (gy) gradients
    int gx = 0, gy = 0;

    // Compute gradients using Sobel operator if pixel is between bounds
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        gx = image[(y - 1) * width + (x + 1)] - image[(y - 1) * width + (x - 1)]
            + 2 * image[y * width + (x + 1)] - 2 * image[y * width + (x - 1)]
            + image[(y + 1) * width + (x + 1)] - image[(y + 1) * width + (x - 1)];
        gy = image[(y + 1) * width + (x - 1)] - image[(y - 1) * width + (x - 1)]
            + 2 * image[(y + 1) * width + x] - 2 * image[(y - 1) * width + x]
            + image[(y + 1) * width + (x + 1)] - image[(y - 1) * width + (x + 1)];
    }

    // Calculate gradient magnitude and map it to a gradient bin
    float gradientMagnitude = sqrtf(gx * gx + gy * gy);
    int gradientBin = min(numBins - 1, (int)(gradientMagnitude / (255.0 / numBins)));

    // Mapping pixel intensity to an intensity bin
    int intensityBin = image[y * width + x] / (256 / numBins);

    // Update histogram using atomicAdd to avoid race conditions
    atomicAdd(&histogram[intensityBin * pitch + gradientBin], 1);
}

// Function to read a grayscale image from file
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

// Function to write histogram data into CSV file
void writeHistogramToCSV(int histogram[][GRADIENT_BINS]) {
    const char* outputPath = "C:\\Users\\ruber\\Downloads\\histogram_output.csv";
    FILE* file = fopen(outputPath, "w");

    if (!file) {
        printf("Error: Could not create output file %s\n", outputPath);
        return;
    }

    // Write CSV headers
    fprintf(file, "Intensity Bin,Gradient Bin,Count\n");

    // Write histogram data
    for (int i = 0; i < INTENSITY_BINS; i++) {
        for (int j = 0; j < GRADIENT_BINS; j++) {
            if (histogram[i][j] > 0) {
                fprintf(file, "%d,%d,%d\n", i, j, histogram[i][j]);
            }
        }
    }

    fclose(file);
    printf("Histogram data saved to: %s\n", outputPath);
}

// CUDA function to compute the histogram
void calculate2DHistogram(const unsigned char* h_image, int width, int height) {
    unsigned char* d_image;
    int* d_histogram;

    size_t imageSize = width * height * sizeof(unsigned char);
    cudaMalloc((void**)&d_image, imageSize);
    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);

    size_t pitch;
    cudaMallocPitch((void**)&d_histogram, &pitch, GRADIENT_BINS * sizeof(int), INTENSITY_BINS);
    cudaMemset(d_histogram, 0, pitch * INTENSITY_BINS);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    compute2DHistogram<<<gridSize, blockSize>>>(d_image, width, height, d_histogram, pitch / sizeof(int), INTENSITY_BINS);

    int (h_histogram)[GRADIENT_BINS] = (int()[GRADIENT_BINS])malloc(INTENSITY_BINS * GRADIENT_BINS * sizeof(int));
    cudaMemcpy2D(h_histogram, GRADIENT_BINS * sizeof(int), d_histogram, pitch, GRADIENT_BINS * sizeof(int), INTENSITY_BINS, cudaMemcpyDeviceToHost);

    writeHistogramToCSV(h_histogram);

    cudaFree(d_image);
    cudaFree(d_histogram);
    free(h_histogram);
}

int main() {
    int width = 305;
    int height = 48;

    const char* filename = "C:\\Users\\ruber\\Downloads\\download.raw";
    unsigned char* image = readRAWImage(filename, width, height);
    if (!image) {
        return -1;
    }

    calculate2DHistogram(image, width, height);

    free(image);
    return 0;
}
