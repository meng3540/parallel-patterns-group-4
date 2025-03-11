#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define histogram dimensions
#define INTENSITY_BINS 256  // Number of bins for intensity
#define GRADIENT_BINS 256   // Number of bins for gradient magnitude

// Kernel to compute gradients and populate the 2D histogram
_global_ void compute2DHistogram(const unsigned char* image, int width, int height,
    int* histogram, int pitch, int numBins) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Compute Sobel gradient (magnitude)
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

    // Normalize gradient to fit into the bins
    int gradientBin = min(numBins - 1, (int)(gradientMagnitude / (255.0 / numBins)));

    // Intensity directly corresponds to the pixel value
    int intensityBin = image[y * width + x] / (256 / numBins);

    // Update the 2D histogram (using atomic operation to avoid race conditions)
    atomicAdd(&histogram[intensityBin * pitch + gradientBin], 1);
}

// Function to write the histogram to a CSV file
void writeHistogramToCSV(const char* filename, int (*histogram)[GRADIENT_BINS]) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Unable to open file %s for writing\n", filename);
        return;
    }

    // Write headers for CSV
    fprintf(file, "Intensity Bin,Gradient Bin,Count\n");

    for (int i = 0; i < INTENSITY_BINS; i++) {
        for (int j = 0; j < GRADIENT_BINS; j++) {
            if (histogram[i][j] > 0) {  // Only write non-zero entries
                fprintf(file, "%d,%d,%d\n", i, j, histogram[i][j]);
            }
        }
    }

    fclose(file);
    printf("Histogram written to %s\n", filename);
}

// Host function to run the kernel
void calculate2DHistogram(const unsigned char* h_image, int width, int height) {
    // Device pointers
    unsigned char* d_image;
    int* d_histogram;

    // Allocate memory for the image on the GPU
    size_t imageSize = width * height * sizeof(unsigned char);
    cudaMalloc((void**)&d_image, imageSize);
    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);

    // Allocate memory for the histogram on the GPU
    size_t pitch;
    cudaMallocPitch((void**)&d_histogram, &pitch, GRADIENT_BINS * sizeof(int), INTENSITY_BINS);
    cudaMemset(d_histogram, 0, pitch * INTENSITY_BINS);

    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    compute2DHistogram << <gridSize, blockSize >> > (d_image, width, height, d_histogram, pitch / sizeof(int), INTENSITY_BINS);

    // Copy the histogram back to host
    int (h_histogram)[GRADIENT_BINS] = (int()[GRADIENT_BINS])malloc(INTENSITY_BINS * GRADIENT_BINS * sizeof(int));
    cudaMemcpy2D(h_histogram, GRADIENT_BINS * sizeof(int), d_histogram, pitch, GRADIENT_BINS * sizeof(int), INTENSITY_BINS, cudaMemcpyDeviceToHost);

    // Write the histogram to a CSV file
    writeHistogramToCSV("C:\\Users\\n01592974\\Downloads\\histogram.csv", h_histogram);
       
    // Free device memory
    cudaFree(d_image);
    cudaFree(d_histogram);

    // Free host memory
    free(h_histogram);
}

// Function to read an image file (grayscale)
// Replace with an actual implementation that reads from a file
unsigned char* readImage(const char* filename, int* width, int* height) {
    *width = 512;  // Example width
    *height = 512; // Example height
    unsigned char* image = (unsigned char*)malloc((*width) * (*height) * sizeof(unsigned char));

    // Fill image with dummy data (replace this with actual file reading logic)
    for (int i = 0; i < (*width) * (*height); i++) {
        image[i] = rand() % 256;
    }

    return image;
}

int main() {
    int width, height;
    unsigned char* image = readImage("image.pgm", &width, &height);

    calculate2DHistogram(image, width, height);

    free(image);
    return 0;

}
