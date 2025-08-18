#include <iostream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <cuda_runtime.h>

// A handy macro for CUDA error checking
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        // CORRECTED LINE: The typo '"' is now " '"
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// Device function to determine texton value from a 2x2 block
__device__ unsigned char casecheck(unsigned char a, unsigned char b, unsigned char c, unsigned char d) {
   if (a == b && b == c && c == d) return 7;
   if (a == b) return 1;
   if (b == d) return 2;
   if (c == d) return 3;
   if (a == c) return 4;
   if (a == d) return 5;
   if (c == b) return 6;
   return 0;
}

// Kernel to calculate the texton image from the value channel
__global__ void calculate_TEXTON_CUDA(const unsigned char* v_channel, unsigned char* texton_img, int rows, int cols)  {
    // Standard 2D global thread index calculation
    int texton_col = blockIdx.x * blockDim.x + threadIdx.x;
    int texton_row = blockIdx.y * blockDim.y + threadIdx.y;
    int texton_cols = cols / 2;
    int texton_rows = rows / 2;

    // Boundary check
    if (texton_col >= texton_cols || texton_row >= texton_rows) {
        return;
    }

    // Map 2D texton index to the top-left corner of the 2x2 block in the source image
    int src_row = texton_row * 2;
    int src_col = texton_col * 2;
    int src_idx = src_row * cols + src_col;

    // Read the 2x2 pixel block
    unsigned char a = v_channel[src_idx];
    unsigned char b = v_channel[src_idx + 1];
    unsigned char c = v_channel[src_idx + cols];
    unsigned char d = v_channel[src_idx + cols + 1];

    // Calculate and write the texton value
    int texton_idx = texton_row * texton_cols + texton_col;
    texton_img[texton_idx] = casecheck(a, b, c, d);
}

// Kernel to calculate Local Texton XOR Pattern (LTxXORp)
__global__ void calculate_LBP_CUDA(const unsigned char* texton_img, unsigned char* lbp_img, int rows, int cols)  {
    // Calculate global thread index for the *interior* of the texton image
    int lbp_col = blockIdx.x * blockDim.x + threadIdx.x;
    int lbp_row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // The grid is for interior points, so the output coordinates are (lbp_row, lbp_col)
    // The corresponding center pixel in the texton_img is at (lbp_row + 1, lbp_col + 1)
    int center_row = lbp_row + 1;
    int center_col = lbp_col + 1;

    // Boundary check (grid is sized for interior, but good practice)
    if (center_row >= rows - 1 || center_col >= cols - 1) {
        return;
    }

    // Index of the center pixel in the source texton image
    int center_idx = center_row * cols + center_col;
    unsigned char center_val = texton_img[center_idx];
    
    // Replicating the original code's specific weights and neighbor logic
    // Your original code used: a=6, b=5, c=4, d=1, e=1, f=4, g=5, h=6
    // The relative offsets for these are complex and depend on width.
    // This implementation uses standard 3x3 neighborhood offsets for correctness and clarity.
    // If your original offsets (a,b,c..) had a different non-local meaning, this logic would need to change.
    const int original_offsets[8] = {
        -cols - 1,  // NW (Top-Left)
        -cols,      // N  (Top-Mid)
        -cols + 1,  // NE (Top-Right)
        -1,         // W  (Left)
        +1,         // E  (Right)
        +cols - 1,  // SW (Bottom-Left)
        +cols,      // S  (Bottom-Mid)
        +cols + 1   // SE (Bottom-Right)
    };
    // Your original weights: wgt[9] = {8, 4, 2,16, 0, 1,32, 64, 128};
    const unsigned int wgt[8] = {8, 4, 2, 16, 1, 32, 64, 128};

    unsigned int lbp_value = 0;
    // Unrolled loop for performance, using your weights
    lbp_value += (texton_img[center_idx + original_offsets[0]] != center_val) * wgt[0]; // NW
    lbp_value += (texton_img[center_idx + original_offsets[1]] != center_val) * wgt[1]; // N
    lbp_value += (texton_img[center_idx + original_offsets[2]] != center_val) * wgt[2]; // NE
    lbp_value += (texton_img[center_idx + original_offsets[3]] != center_val) * wgt[3]; // W
    lbp_value += (texton_img[center_idx + original_offsets[4]] != center_val) * wgt[4]; // E
    lbp_value += (texton_img[center_idx + original_offsets[5]] != center_val) * wgt[5]; // SW
    lbp_value += (texton_img[center_idx + original_offsets[6]] != center_val) * wgt[6]; // S
    lbp_value += (texton_img[center_idx + original_offsets[7]] != center_val) * wgt[7]; // SE

    int lbp_idx = center_row * cols + center_col;
    lbp_img[lbp_idx] = (unsigned char)lbp_value;
}


int main(int argc, char *argv[]) {
    std::cout << "\nCUDA HARDWARE ACTIVATED\n____________________\n" << std::endl;

    // --- 1. Load Image and Prepare Host Data ---
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Couldn't load input image." << std::endl;
        return -1;
    }

    cv::Mat hsv_image, v_channel;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_planes;
    cv::split(hsv_image, hsv_planes);
    v_channel = hsv_planes[2];

    int i_rows = v_channel.rows;
    int i_cols = v_channel.cols;
    int t_rows = i_rows / 2;
    int t_cols = i_cols / 2;

    /*std::cout << "Original V-Channel Image (" << i_rows << "x" << i_cols << "):\n";
    for (int r = 0; r < i_rows; ++r) {
        for (int c = 0; c < i_cols; ++c) {
            printf("%3d ", v_channel.at<unsigned char>(r, c));
        }
        printf("\n");
    }
    printf("\n____________________\n");*/

    // --- 2. Allocate GPU Memory ---
    unsigned char *d_v_channel, *d_texton_img, *d_lbp_img;
    size_t v_size = i_rows * i_cols * sizeof(unsigned char);
    size_t t_size = t_rows * t_cols * sizeof(unsigned char);

    checkCudaErrors(cudaMalloc(&d_v_channel, v_size));
    checkCudaErrors(cudaMalloc(&d_texton_img, t_size));
    checkCudaErrors(cudaMalloc(&d_lbp_img, t_size));

    // --- 3. Copy Data to GPU ---
    checkCudaErrors(cudaMemcpy(d_v_channel, v_channel.data, v_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_lbp_img, 0, t_size));

    // --- 4. Configure Kernel Launches ---
    dim3 threadsPerBlock(16, 16);
    dim3 textonGrid((t_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (t_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 lbpGrid(((t_cols - 2) + threadsPerBlock.x - 1) / threadsPerBlock.x, ((t_rows - 2) + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // --- 5. Execute Kernels and Time with CUDA Events ---
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));

    calculate_TEXTON_CUDA<<<textonGrid, threadsPerBlock>>>(d_v_channel, d_texton_img, i_rows, i_cols);
    calculate_LBP_CUDA<<<lbpGrid, threadsPerBlock>>>(d_texton_img, d_lbp_img, t_rows, t_cols);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop)); 

    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    // --- 6. Copy Results Back to Host ---
    std::vector<unsigned char> h_texton_img(t_rows * t_cols);
    std::vector<unsigned char> h_lbp_img(t_rows * t_cols);
    checkCudaErrors(cudaMemcpy(h_texton_img.data(), d_texton_img, t_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_lbp_img.data(), d_lbp_img, t_size, cudaMemcpyDeviceToHost));
    
    // --- 7. Display Results ---
    /*
    std::cout << "\nTexton image (" << t_rows << "x" << t_cols << "):\n\n";
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols; ++c) {
            printf("%4d", h_texton_img[r * t_cols + c]);
        }
        printf("\n");
    }

    printf("\n___________________\n");
    std::cout << "\nTexton Weight image (LBP) (" << t_rows << "x" << t_cols << "):\n\n";
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols; ++c) {
            printf("%4d", h_lbp_img[r * t_cols + c]);
        }
        printf("\n");
    }*/
    
    printf("\nTotal GPU Kernel Elapsed time: %.5f milliseconds\n", milliseconds);

    // --- 8. Cleanup ---
    checkCudaErrors(cudaFree(d_v_channel));
    checkCudaErrors(cudaFree(d_texton_img));
    checkCudaErrors(cudaFree(d_lbp_img));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}
