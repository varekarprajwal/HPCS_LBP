#include <iostream>
#include <vector>
#include <numeric>
#include <ctime>
#include <mpi.h>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// --- HELPER FUNCTION TO PRINT A CV::MAT ---
void print_matrix(const Mat& mat_to_print, const string& name) {
    if (mat_to_print.empty()) {
        cout << "\n--- " << name << " is empty ---" << endl;
        return;
    }
    cout << "\n--- " << name << " (" << mat_to_print.rows << "x" << mat_to_print.cols << ") ---" << endl;
    for (int i = 0; i < mat_to_print.rows; i++) {
        for (int j = 0; j < mat_to_print.cols; j++) {
            // Use (int) to print the numerical value of the uchar, not the character
            printf("%4d", (int)mat_to_print.at<uchar>(i, j));
        }
        printf("\n");
    }
    cout << "----------------------------------------" << endl;
}

// Same casecheck function, but it now returns a uchar (1 byte)
uchar casecheck(uchar a, uchar b, uchar c, uchar d) {
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
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0 && argc < 2) {
        cerr << "Usage: mpirun -np <N> " << argv[0] << " <image_path>" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    Mat v_channel_full;
    int full_img_rows = 0, full_img_cols = 0;
    int texton_rows = 0, texton_cols = 0;

    // == 1. Root Process: Load Image and Broadcast Dimensions ==
    if (rank == 0) {
        Mat image = imread(argv[1], IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Error: Couldn't load input image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1;
        }

        Mat hsv_image;
        cvtColor(image, hsv_image, COLOR_BGR2HSV);
        vector<Mat> hsv_planes;
        split(hsv_image, hsv_planes);
        v_channel_full = hsv_planes[2];

        full_img_rows = v_channel_full.rows;
        full_img_cols = v_channel_full.cols;
        texton_rows = full_img_rows / 2;
        texton_cols = full_img_cols / 2;
    }

    // Broadcast essential dimensions to all processes
    MPI_Bcast(&full_img_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&full_img_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&texton_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&texton_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // == 2. Distribute V-Channel Image Data ==
    int base_rows_per_proc = (full_img_rows / 2 / world_size) * 2;
    
    vector<int> send_counts(world_size);
    vector<int> displacements(world_size);
    
    if (rank == 0) {
        int current_displacement = 0;
        for (int i = 0; i < world_size; ++i) {
            int rows_to_send = (i < world_size - 1) ? base_rows_per_proc : (full_img_rows - i * base_rows_per_proc);
            send_counts[i] = rows_to_send * full_img_cols;
            displacements[i] = current_displacement;
            current_displacement += send_counts[i];
        }
    }
    
    MPI_Bcast(send_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    int local_v_rows = send_counts[rank] / full_img_cols;
    Mat local_v_channel(local_v_rows, full_img_cols, CV_8UC1);

    MPI_Scatterv(v_channel_full.data, send_counts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
                 local_v_channel.data, send_counts[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // == 3. Parallel Texton Calculation ==
    double startTime1 = MPI_Wtime();
    int local_texton_rows = local_v_rows / 2;
    Mat local_texton(local_texton_rows, texton_cols, CV_8UC1);

    for (int i = 0; i < local_texton_rows; ++i) {
        const uchar* p_src1 = local_v_channel.ptr<uchar>(i * 2);
        const uchar* p_src2 = local_v_channel.ptr<uchar>(i * 2 + 1);
        uchar* p_dest = local_texton.ptr<uchar>(i);
        for (int j = 0; j < texton_cols; ++j) {
            p_dest[j] = casecheck(p_src1[j * 2], p_src1[j * 2 + 1], p_src2[j * 2], p_src2[j * 2 + 1]);
        }
    }
    double endTime1 = MPI_Wtime();

    // == 4. Halo Exchange for LTxXORp ==
    Mat local_texton_with_halos(local_texton_rows + 2, texton_cols, CV_8UC1);
    local_texton.copyTo(local_texton_with_halos(Rect(0, 1, texton_cols, local_texton_rows)));

    int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next_rank = (rank == world_size - 1) ? MPI_PROC_NULL : rank + 1;
    
    MPI_Sendrecv(local_texton.ptr<uchar>(0), texton_cols, MPI_UNSIGNED_CHAR, prev_rank, 0,
                 local_texton_with_halos.ptr<uchar>(0), texton_cols, MPI_UNSIGNED_CHAR, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(local_texton.ptr<uchar>(local_texton_rows - 1), texton_cols, MPI_UNSIGNED_CHAR, next_rank, 0,
                 local_texton_with_halos.ptr<uchar>(local_texton_rows + 1), texton_cols, MPI_UNSIGNED_CHAR, next_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // == 5. Parallel LTxXORp Calculation ==
    double startTime2 = MPI_Wtime();
    Mat local_result = local_texton.clone();
    const int wgt[3][3] = {{8, 4, 2}, {16, 0, 1}, {32, 64, 128}};
    
    int start_row = (rank == 0) ? 1 : 0;
    int end_row = (rank == world_size - 1) ? local_texton_rows - 1 : local_texton_rows;

    for (int i = start_row; i < end_row; ++i) {
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
            p_dest[j] = saturate_cast<uchar>(xor_S);
        }
    }
    double endTime2 = MPI_Wtime();

    // == 6. Gather All Results ==
    vector<int> recv_counts(world_size);
    vector<int> texton_displacements(world_size);
    
    int local_pixel_count = local_texton_rows * texton_cols;
    MPI_Gather(&local_pixel_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    Mat texton_full, final_result_full;
    if (rank == 0) {
        texton_displacements[0] = 0;
        for (size_t i = 1; i < recv_counts.size(); ++i) {
            texton_displacements[i] = texton_displacements[i - 1] + recv_counts[i - 1];
        }
        texton_full.create(texton_rows, texton_cols, CV_8UC1);
        final_result_full.create(texton_rows, texton_cols, CV_8UC1);
    }
    
    MPI_Gatherv(local_texton.data, local_pixel_count, MPI_UNSIGNED_CHAR,
                texton_full.data, recv_counts.data(), texton_displacements.data(),
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
                
    MPI_Gatherv(local_result.data, local_pixel_count, MPI_UNSIGNED_CHAR,
                final_result_full.data, recv_counts.data(), texton_displacements.data(),
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    // == 7. Print Timings and Results on Root ==
    double totalTime1 = (endTime1 - startTime1) * 1000.0;
    double totalTime2 = (endTime2 - startTime2) * 1000.0;
    
    double max_time1, max_time2;
    MPI_Reduce(&totalTime1, &max_time1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalTime2, &max_time2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "\n--- Optimized MPI/OpenCV Execution ---" << endl;
        //print_matrix(texton_full, "Intermediate Texton Image");
        //print_matrix(final_result_full, "Final LTxXORp Result");

        printf("\nMax time for Texton calculation:   %.4f milliseconds\n", max_time1);
        printf("Max time for LTxXORp calculation: %.4f milliseconds\n", max_time2);
        printf("Total max elapsed time:            %.4f milliseconds\n", max_time1 + max_time2);
    }

    MPI_Finalize();
    return 0;
}