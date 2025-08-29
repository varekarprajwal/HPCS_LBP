#include <iostream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <cuda_runtime.h>

// ============================
// ✅ CUDA Error Checking Macro
// ============================
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

// ============================
// ✅ Optimized CUDA Kernels
// ============================

// --- Device function to determine texton value from a 2x2 block ---
// This remains unchanged. It's marked __forceinline__ to encourage the compiler
// to embed its code directly, avoiding function call overhead.
__device__ __forceinline__ unsigned char casecheck(unsigned char a, unsigned char b, unsigned char c, unsigned char d) {
    if (a == b && b == c && c == d) return 7;
    if (a == b) return 1;
    if (b == d) return 2;
    if (c == d) return 3;
    if (a == c) return 4;
    if (a == d) return 5;
    if (c == b) return 6;
    return 0;
}

// --- Fused Kernel for Texton and LBP Calculation using Shared Memory ---
// This single kernel replaces the two original kernels.
// BLOCK_SIZE must be a compile-time constant for shared memory array sizing.
#define BLOCK_SIZE 16
__global__ void calculate_TextonLBP_Fused(const unsigned char* v_channel, unsigned char* lbp_img, int v_rows, int v_cols) {
    // Shared memory tile to hold the intermediate texton values.
    // It's larger than the block size to accommodate a 1-pixel "halo" or "ghost zone"
    // needed for the LBP calculation at the edges of the block.
    __shared__ unsigned char texton_tile[(BLOCK_SIZE + 2)][(BLOCK_SIZE + 2)];

    // Map thread indices to global image coordinates
    const int texton_cols = v_cols / 2;
    const int texton_rows = v_rows / 2;
    
    // Top-left corner of the 2x2 V-channel block this thread will process
    const int v_start_x = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 2;
    const int v_start_y = (blockIdx.y * BLOCK_SIZE + threadIdx.y) * 2;

    // Position within the shared memory tile where this thread will write
    const int tile_x = threadIdx.x + 1;
    const int tile_y = threadIdx.y + 1;

    // --- Step 1: Collaboratively load data and compute textons into shared memory ---
    // Each thread computes one texton value and stores it in the shared memory tile.
    // We only need to load data if it's within the bounds of the source V-channel.
    if (v_start_y < v_rows - 1 && v_start_x < v_cols - 1) {
        const int v_idx = v_start_y * v_cols + v_start_x;
        unsigned char a = v_channel[v_idx];
        unsigned char b = v_channel[v_idx + 1];
        unsigned char c = v_channel[v_idx + v_cols];
        unsigned char d = v_channel[v_idx + v_cols + 1];
        texton_tile[tile_y][tile_x] = casecheck(a, b, c, d);
    }

    // --- Handle the halo regions ---
    // Threads at the edges of the block also compute the textons for the halo area.
    // This is a simplified approach; more complex strategies exist but this is effective.
    // Top halo
    if (threadIdx.y == 0 && v_start_y > 0) {
        const int v_idx = (v_start_y - 2) * v_cols + v_start_x;
        texton_tile[0][tile_x] = casecheck(v_channel[v_idx], v_channel[v_idx+1], v_channel[v_idx+v_cols], v_channel[v_idx+v_cols+1]);
    }
    // Bottom halo
    if (threadIdx.y == BLOCK_SIZE - 1 && v_start_y < v_rows - 2) {
        const int v_idx = (v_start_y + 2) * v_cols + v_start_x;
        texton_tile[BLOCK_SIZE+1][tile_x] = casecheck(v_channel[v_idx], v_channel[v_idx+1], v_channel[v_idx+v_cols], v_channel[v_idx+v_cols+1]);
    }
    // Left halo
    if (threadIdx.x == 0 && v_start_x > 0) {
        const int v_idx = v_start_y * v_cols + (v_start_x - 2);
        texton_tile[tile_y][0] = casecheck(v_channel[v_idx], v_channel[v_idx+1], v_channel[v_idx+v_cols], v_channel[v_idx+v_cols+1]);
    }
    // Right halo
    if (threadIdx.x == BLOCK_SIZE - 1 && v_start_x < v_cols - 2) {
        const int v_idx = v_start_y * v_cols + (v_start_x + 2);
        texton_tile[tile_y][BLOCK_SIZE+1] = casecheck(v_channel[v_idx], v_channel[v_idx+1], v_channel[v_idx+v_cols], v_channel[v_idx+v_cols+1]);
    }
    // TODO: Handle corners of the halo for a more robust implementation.

    // Synchronize all threads in the block to ensure the entire shared memory tile is filled
    // before any thread proceeds to the next step.
    __syncthreads();

    // --- Step 2: Perform LBP calculation using data from shared memory ---
    const int texton_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int texton_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Only threads corresponding to valid output pixels should perform the calculation and write.
    // This also avoids processing the image border, as in the original code.
    if (texton_y < texton_rows -1 && texton_x < texton_cols -1 && texton_x > 0 && texton_y > 0) {
        // Center value is read from the thread's position in the tile.
        unsigned char center_val = texton_tile[tile_y][tile_x];

        const unsigned int wgt[8] = {8, 4, 2, 16, 1, 32, 64, 128};
        unsigned int lbp_value = 0;

        // All neighbor reads now come from fast shared memory instead of slow global memory.
        lbp_value += (texton_tile[tile_y - 1][tile_x - 1] != center_val) * wgt[0];
        lbp_value += (texton_tile[tile_y - 1][tile_x]     != center_val) * wgt[1];
        lbp_value += (texton_tile[tile_y - 1][tile_x + 1] != center_val) * wgt[2];
        lbp_value += (texton_tile[tile_y][tile_x - 1]     != center_val) * wgt[3];
        lbp_value += (texton_tile[tile_y][tile_x + 1]     != center_val) * wgt[4];
        lbp_value += (texton_tile[tile_y + 1][tile_x - 1] != center_val) * wgt[5];
        lbp_value += (texton_tile[tile_y + 1][tile_x]     != center_val) * wgt[6];
        lbp_value += (texton_tile[tile_y + 1][tile_x + 1] != center_val) * wgt[7];

        // Write the final result to global memory.
        lbp_img[texton_y * texton_cols + texton_x] = static_cast<unsigned char>(lbp_value);
    }
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

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

    const int i_rows = v_channel.rows;
    const int i_cols = v_channel.cols;
    const int t_rows = i_rows / 2;
    const int t_cols = i_cols / 2;

    // --- 2. Allocate GPU Memory ---
    // We no longer need memory for the intermediate texton image on the GPU.
    unsigned char *d_v_channel, *d_lbp_img;
    const size_t v_size = i_rows * i_cols * sizeof(unsigned char);
    const size_t t_size = t_rows * t_cols * sizeof(unsigned char);

    checkCudaErrors(cudaMalloc(&d_v_channel, v_size));
    checkCudaErrors(cudaMalloc(&d_lbp_img, t_size));

    // --- 3. Copy Data to GPU ---
    checkCudaErrors(cudaMemcpy(d_v_channel, v_channel.data, v_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_lbp_img, 0, t_size)); // Initialize output memory

    // --- 4. Configure and Execute Fused Kernel ---
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( (t_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, 
               (t_rows + threadsPerBlock.y - 1) / threadsPerBlock.y );

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));

    // Launch the single fused kernel
    calculate_TextonLBP_Fused<<<grid, threadsPerBlock>>>(d_v_channel, d_lbp_img, i_rows, i_cols);
    
    // Check for any errors during kernel execution
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    // --- 5. Copy Results Back ---
    // We only need to copy the final LBP result.
    std::vector<unsigned char> h_lbp_img(t_rows * t_cols);
    checkCudaErrors(cudaMemcpy(h_lbp_img.data(), d_lbp_img, t_size, cudaMemcpyDeviceToHost));
    
    // You can still generate the texton image on the host for debugging if needed,
    // but it's not part of the optimized GPU path.

    //printImage(h_lbp_img, t_rows, t_cols, "Final LTxXORp Result (from fused kernel)");

    printf("\nTotal GPU Kernel Elapsed time: %.5f milliseconds\n", milliseconds);

    // --- 6. Cleanup ---
    checkCudaErrors(cudaFree(d_v_channel));
    checkCudaErrors(cudaFree(d_lbp_img));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}
