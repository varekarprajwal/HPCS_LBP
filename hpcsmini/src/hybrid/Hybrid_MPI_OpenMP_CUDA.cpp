#include <iostream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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
            printf("%4d", static_cast<int>(image[r * cols + c]));
        }
        printf("\n");
    }
}

// ============================
// ✅ Optimized CUDA Kernels
// ============================
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

#define BLOCK_SIZE 16
__global__ void calculate_TextonLBP_Fused(const unsigned char* v_channel, unsigned char* lbp_img, int v_rows, int v_cols) {
    __shared__ unsigned char texton_tile[(BLOCK_SIZE + 2)][(BLOCK_SIZE + 2)];

    const int texton_cols = v_cols / 2;
    const int texton_rows = v_rows / 2;
    
    const int v_start_x = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 2;
    const int v_start_y = (blockIdx.y * BLOCK_SIZE + threadIdx.y) * 2;

    const int tile_x = threadIdx.x + 1;
    const int tile_y = threadIdx.y + 1;

    if (v_start_y < v_rows - 1 && v_start_x < v_cols - 1) {
        const int v_idx = v_start_y * v_cols + v_start_x;
        texton_tile[tile_y][tile_x] = casecheck(v_channel[v_idx], v_channel[v_idx + 1], v_channel[v_idx + v_cols], v_channel[v_idx + v_cols + 1]);
    }

    if (threadIdx.y == 0 && v_start_y > 1) {
        const int v_idx = (v_start_y - 2) * v_cols + v_start_x;
        if (v_idx >= 0) texton_tile[0][tile_x] = casecheck(v_channel[v_idx], v_channel[v_idx+1], v_channel[v_idx+v_cols], v_channel[v_idx+v_cols+1]);
    }
    if (threadIdx.y == BLOCK_SIZE - 1 && v_start_y < v_rows - 3) {
        const int v_idx = (v_start_y + 2) * v_cols + v_start_x;
        texton_tile[BLOCK_SIZE+1][tile_x] = casecheck(v_channel[v_idx], v_channel[v_idx+1], v_channel[v_idx+v_cols], v_channel[v_idx+v_cols+1]);
    }
    if (threadIdx.x == 0 && v_start_x > 1) {
        const int v_idx = v_start_y * v_cols + (v_start_x - 2);
        if (v_idx >=0) texton_tile[tile_y][0] = casecheck(v_channel[v_idx], v_channel[v_idx+1], v_channel[v_idx+v_cols], v_channel[v_idx+v_cols+1]);
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && v_start_x < v_cols - 3) {
        const int v_idx = v_start_y * v_cols + (v_start_x + 2);
        texton_tile[tile_y][BLOCK_SIZE+1] = casecheck(v_channel[v_idx], v_channel[v_idx+1], v_channel[v_idx+v_cols], v_channel[v_idx+v_cols+1]);
    }

    __syncthreads();

    const int texton_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int texton_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (texton_y < texton_rows -1 && texton_x < texton_cols -1 && texton_x > 0 && texton_y > 0) {
        unsigned char center_val = texton_tile[tile_y][tile_x];
        const unsigned int wgt[8] = {8, 4, 2, 16, 1, 32, 64, 128};
        unsigned int lbp_value = 0;
        lbp_value += (texton_tile[tile_y - 1][tile_x - 1] != center_val) * wgt[0];
        lbp_value += (texton_tile[tile_y - 1][tile_x]     != center_val) * wgt[1];
        lbp_value += (texton_tile[tile_y - 1][tile_x + 1] != center_val) * wgt[2];
        lbp_value += (texton_tile[tile_y][tile_x - 1]     != center_val) * wgt[3];
        lbp_value += (texton_tile[tile_y][tile_x + 1]     != center_val) * wgt[4];
        lbp_value += (texton_tile[tile_y + 1][tile_x - 1] != center_val) * wgt[5];
        lbp_value += (texton_tile[tile_y + 1][tile_x]     != center_val) * wgt[6];
        lbp_value += (texton_tile[tile_y + 1][tile_x + 1] != center_val) * wgt[7];
        lbp_img[texton_y * texton_cols + texton_x] = static_cast<unsigned char>(lbp_value);
    }
}

int main(int argc, char *argv[]) {
    // --- 1. Initialize MPI ---
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // --- 2. Select GPU for this MPI process ---
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0) {
        if (rank == 0) std::cerr << "No CUDA-enabled GPUs found." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCudaErrors(cudaSetDevice(rank % num_gpus));

    cv::Mat v_channel_full;
    int i_rows = 0, i_cols = 0;

    // --- 3. Root Process: Load Image and Distribute Data ---
    if (rank == 0) {
        if (argc < 2) {
            std::cerr << "Usage: mpirun -np <N> " << argv[0] << " <image_path>" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Couldn't load input image." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        cv::Mat hsv_image;
        cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsv_planes;
        cv::split(hsv_image, hsv_planes);
        v_channel_full = hsv_planes[2];
        i_rows = v_channel_full.rows;
        i_cols = v_channel_full.cols;
    }

    // Broadcast image dimensions to all processes
    MPI_Bcast(&i_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&i_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- 4. Data Decomposition for Each Process ---
    int rows_per_proc = i_rows / world_size;
    int local_rows = (rank == world_size - 1) ? (i_rows - rank * rows_per_proc) : rows_per_proc;
    
    // Allocate space for local V-channel slice, including halos
    int local_rows_with_halos = local_rows + 2;
    cv::Mat local_v_slice(local_rows_with_halos, i_cols, CV_8UC1);

    // Scatter the main image data
    std::vector<int> send_counts(world_size), displacements(world_size);
    if (rank == 0) {
        for(int i=0; i<world_size; ++i) {
            send_counts[i] = ((i == world_size - 1) ? (i_rows - i * rows_per_proc) : rows_per_proc) * i_cols;
            displacements[i] = (i == 0) ? 0 : displacements[i-1] + send_counts[i-1];
        }
    }
    MPI_Scatterv(v_channel_full.data, send_counts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
                 local_v_slice.data + i_cols, local_rows * i_cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // --- 5. Halo Exchange between MPI processes ---
    int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next_rank = (rank < world_size - 1) ? rank + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(local_v_slice.data + i_cols, i_cols, MPI_UNSIGNED_CHAR, prev_rank, 0,
                 local_v_slice.data + (local_rows + 1) * i_cols, i_cols, MPI_UNSIGNED_CHAR, next_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(local_v_slice.data + local_rows * i_cols, i_cols, MPI_UNSIGNED_CHAR, next_rank, 0,
                 local_v_slice.data, i_cols, MPI_UNSIGNED_CHAR, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // --- 6. Process Slice on GPU ---
    unsigned char *d_v_channel, *d_lbp_img;
    const int t_rows = local_rows / 2;
    const int t_cols = i_cols / 2;
    const size_t v_size = local_rows_with_halos * i_cols * sizeof(unsigned char);
    const size_t t_size = t_rows * t_cols * sizeof(unsigned char);

    checkCudaErrors(cudaMalloc(&d_v_channel, v_size));
    checkCudaErrors(cudaMalloc(&d_lbp_img, t_size));
    checkCudaErrors(cudaMemcpy(d_v_channel, local_v_slice.data, v_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_lbp_img, 0, t_size));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((t_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (t_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    calculate_TextonLBP_Fused<<<grid, threadsPerBlock>>>(d_v_channel, d_lbp_img, local_rows_with_halos, i_cols);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    std::vector<unsigned char> h_lbp_slice(t_rows * t_cols);
    checkCudaErrors(cudaMemcpy(h_lbp_slice.data(), d_lbp_img, t_size, cudaMemcpyDeviceToHost));

    // --- 7. Gather Results ---
    std::vector<unsigned char> h_lbp_full( (i_rows/2) * (i_cols/2) );
    std::vector<int> recv_counts(world_size), recv_displs(world_size);
    if (rank == 0) {
        for(int i=0; i<world_size; ++i) {
            int proc_rows = (i == world_size - 1) ? (i_rows - i * rows_per_proc) : rows_per_proc;
            recv_counts[i] = (proc_rows / 2) * t_cols;
            recv_displs[i] = (i > 0) ? recv_displs[i-1] + recv_counts[i-1] : 0;
        }
    }
    MPI_Gatherv(h_lbp_slice.data(), t_rows * t_cols, MPI_UNSIGNED_CHAR,
                h_lbp_full.data(), recv_counts.data(), recv_displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // --- 8. Finalize and Print Timings ---
    float max_milliseconds;
    MPI_Reduce(&milliseconds, &max_milliseconds, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        //printImage(h_lbp_full, i_rows/2, i_cols/2, "Final LTxXORp Result (Hybrid)");
        printf("\nTotal Max GPU Kernel Elapsed time: %.5f milliseconds\n", max_milliseconds);
    }

    checkCudaErrors(cudaFree(d_v_channel));
    checkCudaErrors(cudaFree(d_lbp_img));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    MPI_Finalize();
    return 0;
}
