#include <iostream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <cuda_runtime.h>

// A handy macro for CUDA error checking
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") at " <<
                  file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// ✅ Generic print function for any numeric type
template <typename T>
void printImage(const std::vector<T>& image, int rows, int cols, const std::string& title) {
    std::cout << "\n" << title << " (" << rows << "x" << cols << "):\n\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%4d", static_cast<int>(image[r * cols + c])); // Cast for display
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

    if (texton_col >= texton_cols || texton_row >= texton_rows) {
        return;
    }

    int src_row = texton_row * 2;
    int src_col = texton_col * 2;
    int src_idx = src_row * cols + src_col;

    unsigned char a = v_channel[src_idx];
    unsigned char b = v_channel[src_idx + 1];
    unsigned char c = v_channel[src_idx + cols];
    unsigned char d = v_channel[src_idx + cols + 1];

    int texton_idx = texton_row * texton_cols + texton_col;
    texton_img[texton_idx] = casecheck(a, b, c, d);
}

// Shared memory optimized LBP kernel
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define TILE_DIM_X (BLOCK_DIM_X + 2)
#define TILE_DIM_Y (BLOCK_DIM_Y + 2)

__global__ void calculate_LBP_CUDA_SharedMem(const unsigned char* texton_img, unsigned char* lbp_img, int rows, int cols) {
    __shared__ unsigned char tile[TILE_DIM_Y][TILE_DIM_X];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_start_x = blockIdx.x * BLOCK_DIM_X;
    int block_start_y = blockIdx.y * BLOCK_DIM_Y;

    int global_row = block_start_y + ty;
    int global_col = block_start_x + tx;

    if (global_row < rows && global_col < cols) {
        tile[ty + 1][tx + 1] = texton_img[global_row * cols + global_col];
    }

    if (ty == 0 && global_row > 0 && global_col < cols) tile[0][tx + 1] = texton_img[(global_row - 1) * cols + global_col];
    if (ty == BLOCK_DIM_Y - 1 && global_row < rows - 1 && global_col < cols) tile[TILE_DIM_Y - 1][tx + 1] = texton_img[(global_row + 1) * cols + global_col];
    if (tx == 0 && global_col > 0 && global_row < rows) tile[ty + 1][0] = texton_img[global_row * cols + (global_col - 1)];
    if (tx == BLOCK_DIM_X - 1 && global_col < cols - 1 && global_row < rows) tile[ty + 1][TILE_DIM_X - 1] = texton_img[global_row * cols + (global_col + 1)];

    if (tx == 0 && ty == 0 && global_row > 0 && global_col > 0) tile[0][0] = texton_img[(global_row - 1) * cols + (global_col - 1)];
    if (tx == BLOCK_DIM_X - 1 && ty == 0 && global_row > 0 && global_col < cols - 1) tile[0][TILE_DIM_X - 1] = texton_img[(global_row - 1) * cols + (global_col + 1)];
    if (tx == 0 && ty == BLOCK_DIM_Y - 1 && global_row < rows - 1 && global_col > 0) tile[TILE_DIM_Y - 1][0] = texton_img[(global_row + 1) * cols + (global_col - 1)];
    if (tx == BLOCK_DIM_X - 1 && ty == BLOCK_DIM_Y - 1 && global_row < rows - 1 && global_col < cols - 1) tile[TILE_DIM_Y - 1][TILE_DIM_X - 1] = texton_img[(global_row + 1) * cols + (global_col + 1)];

    __syncthreads();

    if (global_row > 0 && global_row < rows - 1 && global_col > 0 && global_col < cols - 1) {
        unsigned char center_val = tile[ty + 1][tx + 1];
        const unsigned int wgt[8] = {8, 4, 2, 16, 1, 32, 64, 128};
        unsigned int lbp_value = 0;

        lbp_value += (tile[ty][tx] != center_val) * wgt[0];
        lbp_value += (tile[ty][tx + 1] != center_val) * wgt[1];
        lbp_value += (tile[ty][tx + 2] != center_val) * wgt[2];
        lbp_value += (tile[ty + 1][tx] != center_val) * wgt[3];
        lbp_value += (tile[ty + 1][tx + 2] != center_val) * wgt[4];
        lbp_value += (tile[ty + 2][tx] != center_val) * wgt[5];
        lbp_value += (tile[ty + 2][tx + 1] != center_val) * wgt[6];
        lbp_value += (tile[ty + 2][tx + 2] != center_val) * wgt[7];

        lbp_img[global_row * cols + global_col] = (unsigned char)lbp_value;
    }
}

int main(int argc, char *argv[]) {
    std::cout << "\nCUDA HARDWARE ACTIVATED\n____________________\n" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

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

    unsigned char *d_v_channel, *d_texton_img, *d_lbp_img;
    size_t v_size = i_rows * i_cols * sizeof(unsigned char);
    size_t t_size = t_rows * t_cols * sizeof(unsigned char);

    checkCudaErrors(cudaMalloc(&d_v_channel, v_size));
    checkCudaErrors(cudaMalloc(&d_texton_img, t_size));
    checkCudaErrors(cudaMalloc(&d_lbp_img, t_size));

    checkCudaErrors(cudaMemcpy(d_v_channel, v_channel.data, v_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_lbp_img, 0, t_size));

    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 textonGrid((t_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (t_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 lbpGrid((t_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (t_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));

    calculate_TEXTON_CUDA<<<textonGrid, threadsPerBlock>>>(d_v_channel, d_texton_img, i_rows, i_cols);
    calculate_LBP_CUDA_SharedMem<<<lbpGrid, threadsPerBlock>>>(d_texton_img, d_lbp_img, t_rows, t_cols);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    std::vector<unsigned char> h_texton_img(t_rows * t_cols);
    std::vector<unsigned char> h_lbp_img(t_rows * t_cols);
    checkCudaErrors(cudaMemcpy(h_texton_img.data(), d_texton_img, t_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_lbp_img.data(), d_lbp_img, t_size, cudaMemcpyDeviceToHost));

    //printImage(h_texton_img, t_rows, t_cols, "Texton image");
    //printf("\n___________________\n");
    //printImage(h_lbp_img, t_rows, t_cols, "Texton Weight image (LBP)");

    printf("\nTotal GPU Kernel Elapsed time: %.5f milliseconds\n", milliseconds);

    checkCudaErrors(cudaFree(d_v_channel));
    checkCudaErrors(cudaFree(d_texton_img));
    checkCudaErrors(cudaFree(d_lbp_img));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}
