#include <iostream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <cuda_runtime.h>

// CUDA error checking macro
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " (" << cudaGetErrorString(result) << ") at "
                  << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// ============================
// ✅ Generic Image Print Function
// ============================
template <typename T>
void printImage(const std::vector<T>& image, int rows, int cols, const std::string& title) {
    std::cout << "\n" << title << " (" << rows << "x" << cols << "):\n\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%4d", static_cast<int>(image[r * cols + c])); // Casting for display
        }
        printf("\n");
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
__global__ void calculate_TEXTON_CUDA(const unsigned char* v_channel, unsigned char* texton_img, int rows, int cols) {
    int texton_col = blockIdx.x * blockDim.x + threadIdx.x;
    int texton_row = blockIdx.y * blockDim.y + threadIdx.y;
    int texton_cols = cols / 2;
    int texton_rows = rows / 2;

    if (texton_col >= texton_cols || texton_row >= texton_rows) return;

    int src_row = texton_row * 2;
    int src_col = texton_col * 2;
    int src_idx = src_row * cols + src_col;
    /// start time

    unsigned char a = v_channel[src_idx];
    unsigned char b = v_channel[src_idx + 1];
    unsigned char c = v_channel[src_idx + cols];
    unsigned char d = v_channel[src_idx + cols + 1];

    int texton_idx = texton_row * texton_cols + texton_col;
    texton_img[texton_idx] = casecheck(a, b, c, d);
    //// end time
}

// Kernel to calculate Local Texton XOR Pattern (LTxXORp)
__global__ void calculate_LBP_CUDA(const unsigned char* texton_img, unsigned char* lbp_img, int rows, int cols) {
    int lbp_col = blockIdx.x * blockDim.x + threadIdx.x;
    int lbp_row = blockIdx.y * blockDim.y + threadIdx.y;

    int center_row = lbp_row + 1;
    int center_col = lbp_col + 1;

    if (center_row >= rows - 1 || center_col >= cols - 1) return;

    int center_idx = center_row * cols + center_col;
    unsigned char center_val = texton_img[center_idx];

    const int original_offsets[8] = {
        -cols - 1, -cols, -cols + 1,
        -1, +1,
        +cols - 1, +cols, +cols + 1
    };
    const unsigned int wgt[8] = {8, 4, 2, 16, 1, 32, 64, 128};

    unsigned int lbp_value = 0;
    /// start time
    lbp_value += (texton_img[center_idx + original_offsets[0]] != center_val) * wgt[0];
    lbp_value += (texton_img[center_idx + original_offsets[1]] != center_val) * wgt[1];
    lbp_value += (texton_img[center_idx + original_offsets[2]] != center_val) * wgt[2];
    lbp_value += (texton_img[center_idx + original_offsets[3]] != center_val) * wgt[3];
    lbp_value += (texton_img[center_idx + original_offsets[4]] != center_val) * wgt[4];
    lbp_value += (texton_img[center_idx + original_offsets[5]] != center_val) * wgt[5];
    lbp_value += (texton_img[center_idx + original_offsets[6]] != center_val) * wgt[6];
    lbp_value += (texton_img[center_idx + original_offsets[7]] != center_val) * wgt[7];

    int lbp_idx = center_row * cols + center_col;
    lbp_img[lbp_idx] = (unsigned char)lbp_value;
    //// end time
}

int main(int argc, char *argv[]) {
    //std::cout << "\nCUDA HARDWARE ACTIVATED\n____________________\n" << std::endl;

    // --- 1. Load Image ---
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

    // --- 4. Configure Kernels ---
    dim3 threadsPerBlock(16, 16);
    dim3 textonGrid((t_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (t_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 lbpGrid(((t_cols - 2) + threadsPerBlock.x - 1) / threadsPerBlock.x, ((t_rows - 2) + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // --- 5. Execute Kernels and Measure Time ---
    cudaEvent_t start_texton, stop_texton;
    checkCudaErrors(cudaEventCreate(&start_texton));
    checkCudaErrors(cudaEventCreate(&stop_texton));

    // Measure time for calculate_TEXTON_CUDA kernel
    checkCudaErrors(cudaEventRecord(start_texton));
    calculate_TEXTON_CUDA<<<textonGrid, threadsPerBlock>>>(d_v_channel, d_texton_img, i_rows, i_cols);
    checkCudaErrors(cudaEventRecord(stop_texton));
    checkCudaErrors(cudaEventSynchronize(stop_texton));

    float texton_milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&texton_milliseconds, start_texton, stop_texton));
    printf("Texton Kernel Elapsed time: %.5f milliseconds\n", texton_milliseconds);

    // Now, measure time for the second kernel
    cudaEvent_t start_lbp, stop_lbp;
    checkCudaErrors(cudaEventCreate(&start_lbp));
    checkCudaErrors(cudaEventCreate(&stop_lbp));

    checkCudaErrors(cudaEventRecord(start_lbp));
    calculate_LBP_CUDA<<<lbpGrid, threadsPerBlock>>>(d_texton_img, d_lbp_img, t_rows, t_cols);
    checkCudaErrors(cudaEventRecord(stop_lbp));
    checkCudaErrors(cudaEventSynchronize(stop_lbp));

    float lbp_milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&lbp_milliseconds, start_lbp, stop_lbp));
    printf("LBP Kernel Elapsed time: %.5f milliseconds\n", lbp_milliseconds);
    printf("Total Elapsed time: %.5f milliseconds\n", texton_milliseconds + lbp_milliseconds);

    // --- 6. Copy Results Back ---
    std::vector<unsigned char> h_texton_img(t_rows * t_cols);
    std::vector<unsigned char> h_lbp_img(t_rows * t_cols);
    checkCudaErrors(cudaMemcpy(h_texton_img.data(), d_texton_img, t_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_lbp_img.data(), d_lbp_img, t_size, cudaMemcpyDeviceToHost));

    // ✅ Use the generic print function instead of commented code
    //printImage(h_texton_img, t_rows, t_cols, "Texton image");
    //std::cout << "\n___________________\n";
    //printImage(h_lbp_img, t_rows, t_cols, "Texton Weight image (LBP)");

    // --- 7. Cleanup ---
    checkCudaErrors(cudaFree(d_v_channel));
    checkCudaErrors(cudaFree(d_texton_img));
    checkCudaErrors(cudaFree(d_lbp_img));
    checkCudaErrors(cudaEventDestroy(start_texton));
    checkCudaErrors(cudaEventDestroy(stop_texton));
    checkCudaErrors(cudaEventDestroy(start_lbp));
    checkCudaErrors(cudaEventDestroy(stop_lbp));

    return 0;
}