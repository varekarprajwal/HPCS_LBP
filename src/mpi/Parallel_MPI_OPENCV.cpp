#include <iostream>
#include <vector>
#include <numeric>
#include <mpi.h>
#include <opencv4/opencv2/opencv.hpp>

// --- HELPER FUNCTION TO PRINT A CV::MAT ---
// A utility function for debugging to print the contents of an OpenCV Mat.
void print_matrix(const cv::Mat& mat_to_print, const std::string& name) {
    if (mat_to_print.empty()) {
        std::cout << "\n--- " << name << " is empty ---" << std::endl;
        return;
    }
    std::cout << "\n--- " << name << " (" << mat_to_print.rows << "x" << mat_to_print.cols << ") ---" << std::endl;
    for (int i = 0; i < mat_to_print.rows; i++) {
        for (int j = 0; j < mat_to_print.cols; j++) {
            // Use (int) to print the numerical value of the uchar, not the character.
            printf("%4d", static_cast<int>(mat_to_print.at<uchar>(i, j)));
        }
        printf("\n");
    }
    std::cout << "----------------------------------------" << std::endl;
}

// --- CASECHECK FUNCTION ---
// This function remains the same, but it's good practice to mark it as inline
// for potential performance gains in the main loop.
inline uchar casecheck(uchar a, uchar b, uchar c, uchar d) {
    if (a == b && b == c && c == d) return 7;
    if (a == b) return 1;
    if (b == d) return 2;
    if (c == d) return 3;
    if (a == c) return 4;
    if (a == d) return 5;
    if (c == b) return 6;
    return 0;
}

int main(int argc, char* argv[]) {
    // --- 1. MPI Initialization ---
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // --- 2. Root Process: Argument & Image Loading ---
    cv::Mat v_channel_full;
    int full_img_rows = 0, full_img_cols = 0;
    int texton_rows = 0, texton_cols = 0;

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

        full_img_rows = v_channel_full.rows;
        full_img_cols = v_channel_full.cols;
        // Ensure dimensions are even for 2x2 processing
        if (full_img_rows % 2 != 0 || full_img_cols % 2 != 0) {
            std::cerr << "Error: Image dimensions must be even." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        texton_rows = full_img_rows / 2;
        texton_cols = full_img_cols / 2;
    }
    

    // --- 3. Broadcast Image Dimensions ---
    // Broadcast dimensions to all processes.
    MPI_Bcast(&full_img_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&full_img_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&texton_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&texton_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- 4. Distribute V-Channel Data ---
    // Calculate how many rows of the original V-channel each process gets.
    // This logic ensures that each process gets an even number of rows for the 2x2 texton calculation.
    std::vector<int> send_counts(world_size);
    std::vector<int> displacements(world_size);

    if (rank == 0) {
        int rows_per_proc = (texton_rows / world_size) * 2;
        int current_displacement = 0;
        for (int i = 0; i < world_size; ++i) {
            int rows_to_send = (i < world_size - 1) ? rows_per_proc : (full_img_rows - i * rows_per_proc);
            send_counts[i] = rows_to_send * full_img_cols;
            displacements[i] = current_displacement;
            current_displacement += send_counts[i];
        }
    }
    
    // Scatter the pixel counts to all processes.
    int local_v_pixel_count = 0;
    MPI_Scatter(send_counts.data(), 1, MPI_INT, &local_v_pixel_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int local_v_rows = local_v_pixel_count / full_img_cols;
    cv::Mat local_v_channel(local_v_rows, full_img_cols, CV_8UC1);

    // Scatter the actual image data from the root process to all processes.
    // The recvcount parameter is now correct, using the specific count for this process.
    MPI_Scatterv(v_channel_full.data, send_counts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
                 local_v_channel.data, local_v_pixel_count, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // --- 5. Parallel Texton Calculation ---
    int local_texton_rows = local_v_rows / 2;
    cv::Mat local_texton(local_texton_rows, texton_cols, CV_8UC1);

    double startTime1 = MPI_Wtime();
    for (int i = 0; i < local_texton_rows; ++i) {
        const uchar* p_src1 = local_v_channel.ptr<uchar>(2 * i);
        const uchar* p_src2 = local_v_channel.ptr<uchar>(2 * i + 1);
        uchar* p_dest = local_texton.ptr<uchar>(i);
        for (int j = 0; j < texton_cols; ++j) {
            p_dest[j] = casecheck(p_src1[2 * j], p_src1[2 * j + 1], p_src2[2 * j], p_src2[2 * j + 1]);
        }
    }
    double endTime1 = MPI_Wtime();

    // --- 6. Halo Exchange for LTxXORp ---
    // Create a larger matrix to hold the local data plus one row above and one below (halos).
    cv::Mat local_texton_with_halos(local_texton_rows + 2, texton_cols, CV_8UC1);
    local_texton.copyTo(local_texton_with_halos(cv::Rect(0, 1, texton_cols, local_texton_rows)));

    int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next_rank = (rank == world_size - 1) ? MPI_PROC_NULL : rank + 1;

    // Send the top row to the previous process and receive its bottom row.
    MPI_Sendrecv(local_texton.ptr<uchar>(0), texton_cols, MPI_UNSIGNED_CHAR, prev_rank, 0,
                 local_texton_with_halos.ptr<uchar>(0), texton_cols, MPI_UNSIGNED_CHAR, prev_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send the bottom row to the next process and receive its top row.
    MPI_Sendrecv(local_texton.ptr<uchar>(local_texton_rows - 1), texton_cols, MPI_UNSIGNED_CHAR, next_rank, 0,
                 local_texton_with_halos.ptr<uchar>(local_texton_rows + 1), texton_cols, MPI_UNSIGNED_CHAR, next_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // --- 7. Parallel LTxXORp Calculation ---
    cv::Mat local_result = local_texton.clone();
    const int wgt[3][3] = {{8, 4, 2}, {16, 0, 1}, {32, 64, 128}};
    
    // Determine the rows this process is responsible for, avoiding the borders unless it's the first or last process.
    int start_row = (rank == 0) ? 1 : 0;
    int end_row = (rank == world_size - 1) ? local_texton_rows - 1 : local_texton_rows;

    double startTime2 = MPI_Wtime();
    for (int i = start_row; i < end_row; ++i) {
        // Pointers to the previous, current, and next rows in the halo-exchanged matrix.
        const uchar* p_prev = local_texton_with_halos.ptr<uchar>(i);
        const uchar* p_curr = local_texton_with_halos.ptr<uchar>(i + 1);
        const uchar* p_next = local_texton_with_halos.ptr<uchar>(i + 2);
        uchar* p_dest = local_result.ptr<uchar>(i);

        for (int j = 1; j < texton_cols - 1; ++j) {
            uchar center_val = p_curr[j];
            int xor_S = 0;
            xor_S += (p_prev[j - 1] != center_val) * wgt[0][0];
            xor_S += (p_prev[j]     != center_val) * wgt[0][1];
            xor_S += (p_prev[j + 1] != center_val) * wgt[0][2];
            xor_S += (p_curr[j - 1] != center_val) * wgt[1][0];
            xor_S += (p_curr[j + 1] != center_val) * wgt[1][2];
            xor_S += (p_next[j - 1] != center_val) * wgt[2][0];
            xor_S += (p_next[j]     != center_val) * wgt[2][1];
            xor_S += (p_next[j + 1] != center_val) * wgt[2][2];
            p_dest[j] = cv::saturate_cast<uchar>(xor_S);
        }
    }
    double endTime2 = MPI_Wtime();

    // --- 8. Gather All Results ---
    int local_pixel_count = local_texton_rows * texton_cols;
    std::vector<int> recv_counts(world_size);
    std::vector<int> texton_displacements(world_size);
    
    // Gather the number of pixels each process handled.
    MPI_Gather(&local_pixel_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    cv::Mat texton_full, final_result_full;
    if (rank == 0) {
        texton_displacements[0] = 0;
        for (size_t i = 1; i < recv_counts.size(); ++i) {
            texton_displacements[i] = texton_displacements[i - 1] + recv_counts[i - 1];
        }
        texton_full.create(texton_rows, texton_cols, CV_8UC1);
        final_result_full.create(texton_rows, texton_cols, CV_8UC1);
    }
    
    // Gather the local texton and final result data into the full matrices on the root process.
    MPI_Gatherv(local_texton.data, local_pixel_count, MPI_UNSIGNED_CHAR,
                texton_full.data, recv_counts.data(), texton_displacements.data(),
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
                
    MPI_Gatherv(local_result.data, local_pixel_count, MPI_UNSIGNED_CHAR,
                final_result_full.data, recv_counts.data(), texton_displacements.data(),
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    // --- 9. Finalize and Print Timings ---
    double totalTime1 = (endTime1 - startTime1) * 1000.0;
    double totalTime2 = (endTime2 - startTime2) * 1000.0;
    
    double max_time1, max_time2;
    MPI_Reduce(&totalTime1, &max_time1, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalTime2, &max_time2, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    // Add a barrier to synchronize all processes before printing.
    // This helps prevent output from different processes from interleaving.
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n--- Optimized MPI/OpenCV Execution ---" << std::endl;
        // Uncomment to see the intermediate and final images.
        //print_matrix(texton_full, "Intermediate Texton Image");
        //print_matrix(final_result_full, "Final LTxXORp Result");

        printf("\nMax time for Texton calculation   :%.4f milliseconds\n", max_time1);
        printf("Max time for LTxXORp calculation  :%.4f milliseconds\n", max_time2);
        printf("Total max elapsed time            :%.4f milliseconds\n", max_time1 + max_time2);
    }

    MPI_Finalize();
    return 0;
}