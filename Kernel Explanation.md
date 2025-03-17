
CUDA Kernel Explanation

The compute2DHistogram kernel is designed to show a 2D histogram based on two 
factors Intensity and Gradient magnitude of a grayscale image. It is executed
with a grid of thread blocks, where each thread is responsible for 
processing a pixel from the image. The grid and block dimensions are calculated 
in the main function to ensure that all pixels are covered.

Step 1: Identifying the Pixel's Position

Each thread calculates its position (x, y) in the image grid using the formulas:

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

Each thread must identify its corresponding pixel. The subsequent boundary 
condition check:

    if (x >= width || y >= height) return;

Prevents memory errors.

Step 2: Computing Gradients

The kernel calculates gx and gy gradients using 
Sobel operator. The Sobel operator is a convolution 
filter used for edge detection.
The logic applied here is:

    gx = image[(y - 1) * width + (x + 1)] - image[(y - 1) * width + (x - 1)]
        + 2 * image[y * width + (x + 1)] - 2 * image[y * width + (x - 1)]
        + image[(y + 1) * width + (x + 1)] - image[(y + 1) * width + (x - 1)];

It combines pixel intensity differences in both horizontal 
and vertical directions, emphasizing edge changes.

Step 3: Computing Magnitude and Binning

The gradient magnitude is calculated with:

    float gradientMagnitude = sqrtf(gx * gx + gy * gy);

This represents the strength of the gradient at each 
pixel. The gradient magnitude is then mapped to a 
specific bin:

    int gradientBin = min(numBins - 1, (int)(gradientMagnitude / (255.0 / numBins)));

Gradient values are divided into predefined bin ranges.

The intensity bin is determined by dividing the pixel intensity by the number
of bins:

    int intensityBin = image[y * width + x] / (256 / numBins);

This effectively maps the pixel's intensity to its bin for histogram generation.

Step 4: Updating the Histogram

To ensure safe parallel writes, the code uses an atomic operation:

    atomicAdd(&histogram[intensityBin * pitch + gradientBin], 1);

The atomicAdd function ensures that multiple threads increment the same histogram 
bin safely without race conditions, maintaining accurate counts.

Overall Functionality

This kernel efficiently calculates a 2D histogram by combining intensity and gradient
data, allowing image analysis. It leverages CUDA's parallel capabilities, 
this method is optimized for large-scale image processing.

