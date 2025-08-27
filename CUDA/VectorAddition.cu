#include <iostream>
#include <vector>
#include <cstdlib> // For rand()

// --- CUDA Error Checking Utility ---
// A simple macro to wrap CUDA API calls and check for errors.
// This is a good practice for any CUDA program.
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        // If a CUDA error occurs, print it to the console and exit.
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) 
                  << " at " << file << ":" << line 
                  << " '" << func << "' \n";
        cudaDeviceReset(); // Resets the device to a clean state.
        exit(99);
    }
}


// --- CUDA Kernel Definition ---
// The `__global__` keyword indicates that this function runs on the device (GPU)
// and can be called from the host (CPU). This is our kernel.
__global__ void addVectors(float* A, float* B, float* C, int n) {
    // Calculate the global index of the current thread.
    // Each thread gets a unique index to work on a different element of the vectors.
    // - threadIdx.x: The thread's index within its block.
    // - blockDim.x: The number of threads in each block.
    // - blockIdx.x: The block's index within the grid.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // A common "guard" condition. Since we might launch more threads than
    // there are elements in the vector (for alignment reasons), we must
    // ensure a thread only works on valid data.
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


// --- Main Host Function ---
int main() {
    // 1. --- Host-side Data Setup ---
    int N = 1024; // The number of elements in our vectors.
    size_t size = N * sizeof(float);

    // Allocate memory for our vectors on the host (CPU RAM).
    // We use std::vector for easier memory management on the host.
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);

    // Initialize the host vectors with some random data.
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand() % 100);
        h_B[i] = static_cast<float>(rand() % 100);
    }

    std::cout << "Step 1: Host data initialized." << std::endl;

    // 2. --- Device-side Memory Allocation ---
    // Pointers for our vectors on the device (GPU VRAM).
    float *d_A, *d_B, *d_C;

    // Allocate memory on the GPU for each vector.
    checkCudaErrors(cudaMalloc(&d_A, size));
    checkCudaErrors(cudaMalloc(&d_B, size));
    checkCudaErrors(cudaMalloc(&d_C, size));

    std::cout << "Step 2: Device memory allocated." << std::endl;


    // 3. --- Copy Data from Host to Device ---
    // Copy the contents of h_A and h_B to d_A and d_B on the GPU.
    // cudaMemcpyHostToDevice specifies the direction of the copy.
    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    std::cout << "Step 3: Data copied from Host to Device." << std::endl;


    // 4. --- Launch the Kernel ---
    // Define the execution configuration for the kernel launch.
    // We'll launch a grid of blocks, with a certain number of threads per block.
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed in the grid to cover all N elements.
    // The formula (N + threadsPerBlock - 1) / threadsPerBlock is a standard
    // way to calculate the ceiling of N / threadsPerBlock.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Launching kernel with " << blocksPerGrid << " blocks and " 
              << threadsPerBlock << " threads per block." << std::endl;

    // Launch the addVectors kernel on the device.
    // The <<<...>>> syntax is how we specify the execution configuration.
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for any errors that might have occurred during the kernel launch.
    checkCudaErrors(cudaGetLastError());
    // Wait for the GPU to finish its work before proceeding.
    checkCudaErrors(cudaDeviceSynchronize());
    
    std::cout << "Step 4: Kernel execution finished." << std::endl;


    // 5. --- Copy Data from Device to Host ---
    // Copy the result vector d_C from the GPU back to the host vector h_C.
    // cudaMemcpyDeviceToHost specifies the direction of the copy.
    checkCudaErrors(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    std::cout << "Step 5: Result copied from Device to Host." << std::endl;


    // 6. --- Verification ---
    // Check the results on the host to make sure the GPU did the right thing.
    bool success = true;
    for (int i = 0; i < N; ++i) {
        // Use a small tolerance for floating-point comparisons.
        if (abs((h_A[i] + h_B[i]) - h_C[i]) > 1e-5) {
            std::cout << "Error at index " << i << ": " 
                      << h_A[i] << " + " << h_B[i] << " = " << h_A[i] + h_B[i] 
                      << ", but GPU result is " << h_C[i] << std::endl;
            success = false;
            break;
        }
    }
    std::cout << "Step 6: Verification " << (success ? "PASSED" : "FAILED") << "." << std::endl;


    // 7. --- Cleanup ---
    // Free the memory that was allocated on the GPU.
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    std::cout << "Step 7: Device memory freed. Program finished." << std::endl;

    return 0;
}

