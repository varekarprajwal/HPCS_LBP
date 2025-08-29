#include <iostream>
#include <vector>
#include <numeric>
#include <mpi.h>
#include <opencv4/opencv2/opencv.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cv;
using namespace std;

// --- HELPER FUNCTION TO PRINT A CV::MAT (debug only) ---
void print_matrix(const Mat& mat_to_print, const string& name) {
    if (mat_to_print.empty()) {
        cout << "\n--- " << name << " is empty ---" << endl;
        return;
    }
    cout << "\n--- " << name << " (" << mat_to_print.rows << "x" << mat_to_print.cols << ") ---" << endl;
    for (int i = 0; i < mat_to_print.rows; i++) {
        for (int j = 0; j < mat_to_print.cols; j++) {
            printf("%4d", (int)mat_to_print.at<uchar>(i, j));
        }
        printf("\n");
    }
    cout << "----------------------------------------" << endl;
}

// --- CASECHECK FUNCTION ---
inline uchar casecheck_fast(uchar a, uchar b, uchar c, uchar d) {
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
    // --- Phase 0: MPI Initialization & Parameter Setup ---
    MPI_Init(&argc, &argv);
    double total_start_time = MPI_Wtime();

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0 && argc < 2) {
        cerr << "Usage: mpirun -np <N> " << argv[0] << " <image_path>" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Mat v_channel_full;
    int full_img_rows = 0, full_img_cols = 0;
    int texton_rows_full = 0, texton_cols_full = 0;

    // Root loads image and computes sizes
    if (rank == 0) {
        Mat image = imread(argv[1], IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Error: Couldn't load input image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        Mat hsv_image;
        cvtColor(image, hsv_image, COLOR_BGR2HSV);
        vector<Mat> hsv_planes;
        split(hsv_image, hsv_planes);
        v_channel_full = hsv_planes[2];

        full_img_rows = v_channel_full.rows;
        full_img_cols = v_channel_full.cols;
        texton_rows_full = full_img_rows / 2;
        texton_cols_full = full_img_cols / 2;
    }

    // Broadcast dimensions to all ranks
    MPI_Bcast(&full_img_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&full_img_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&texton_rows_full, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&texton_cols_full, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Phase 1: Corrected Data Distribution ---
    // Distribute based on TEXTON rows to ensure each rank gets an even number of V-channel rows.
    int base_texton_rows = texton_rows_full / world_size;
    int remainder_texton_rows = texton_rows_full % world_size;
    int local_texton_rows = base_texton_rows + (rank < remainder_texton_rows ? 1 : 0);
    int local_v_rows = local_texton_rows * 2;

    vector<int> v_send_counts(world_size);
    vector<int> v_displacements(world_size);
    vector<int> t_recv_counts(world_size);
    vector<int> t_displacements(world_size);

    if (rank == 0) {
        int v_disp = 0, t_disp = 0;
        for (int i = 0; i < world_size; ++i) {
            int t_rows = base_texton_rows + (i < remainder_texton_rows ? 1 : 0);
            int v_rows = t_rows * 2;

            v_send_counts[i] = v_rows * full_img_cols;
            v_displacements[i] = v_disp;
            v_disp += v_send_counts[i];

            t_recv_counts[i] = t_rows * texton_cols_full;
            t_displacements[i] = t_disp;
            t_disp += t_recv_counts[i];
        }
    }

    // Allocate local buffer and scatter V-channel data
    Mat local_v_channel(local_v_rows, full_img_cols, CV_8UC1);
    MPI_Scatterv(v_channel_full.data, v_send_counts.data(), v_displacements.data(),
                 MPI_UNSIGNED_CHAR, local_v_channel.data, local_v_rows * full_img_cols,
                 MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // --- Phase 2: Local Texton Calculation (OpenMP) ---
    double texton_start_time = MPI_Wtime();
    Mat local_texton(local_texton_rows, texton_cols_full, CV_8UC1);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < local_texton_rows; ++i) {
        const uchar* p_top = local_v_channel.ptr<uchar>(i * 2);
        const uchar* p_bot = local_v_channel.ptr<uchar>(i * 2 + 1);
        uchar* p_dst = local_texton.ptr<uchar>(i);
        for (int j = 0; j < texton_cols_full; ++j) {
            p_dst[j] = casecheck_fast(p_top[j * 2], p_top[j * 2 + 1], p_bot[j * 2], p_bot[j * 2 + 1]);
        }
    }
    double texton_end_time = MPI_Wtime();

    // --- Phase 3: Halo Exchange ---
    double ltxxorp_start_time = MPI_Wtime();
    int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next_rank = (rank == world_size - 1) ? MPI_PROC_NULL : rank + 1;

    Mat halo_top(1, texton_cols_full, CV_8UC1);
    Mat halo_bottom(1, texton_cols_full, CV_8UC1);

    // Non-blocking send and receive for halo exchange
    MPI_Request reqs[4];
    int req_count = 0;
    if (prev_rank != MPI_PROC_NULL) {
        MPI_Irecv(halo_top.data, texton_cols_full, MPI_UNSIGNED_CHAR, prev_rank, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(local_texton.ptr(0), texton_cols_full, MPI_UNSIGNED_CHAR, prev_rank, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    if (next_rank != MPI_PROC_NULL) {
        MPI_Irecv(halo_bottom.data, texton_cols_full, MPI_UNSIGNED_CHAR, next_rank, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(local_texton.ptr(local_texton_rows - 1), texton_cols_full, MPI_UNSIGNED_CHAR, next_rank, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // --- Phase 4: Final LTxXORp Computation (OpenMP) ---
    Mat local_result(local_texton_rows, texton_cols_full, CV_8UC1, Scalar(0));
    const int wgt[3][3] = {{8, 4, 2}, {16, 0, 1}, {32, 64, 128}};

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < local_texton_rows; ++i) {
        uchar* p_dest = local_result.ptr<uchar>(i);
        for (int j = 1; j < texton_cols_full - 1; ++j) {
            const uchar* p_prev = (i == 0) ? halo_top.ptr<uchar>(0) : local_texton.ptr<uchar>(i - 1);
            const uchar* p_curr = local_texton.ptr<uchar>(i);
            const uchar* p_next = (i == local_texton_rows - 1) ? halo_bottom.ptr<uchar>(0) : local_texton.ptr<uchar>(i + 1);
            
            uchar center = p_curr[j];
            int xor_S = (p_prev[j - 1] != center) * wgt[0][0]
                      + (p_prev[j]     != center) * wgt[0][1]
                      + (p_prev[j + 1] != center) * wgt[0][2]
                      + (p_curr[j - 1] != center) * wgt[1][0]
                      + (p_curr[j + 1] != center) * wgt[1][2]
                      + (p_next[j - 1] != center) * wgt[2][0]
                      + (p_next[j]     != center) * wgt[2][1]
                      + (p_next[j + 1] != center) * wgt[2][2];
            p_dest[j] = saturate_cast<uchar>(xor_S);
        }
    }
    double ltxxorp_end_time = MPI_Wtime();

    // --- Phase 5: Result Gathering ---
    Mat texton_full, final_result_full;
    if (rank == 0) {
        texton_full.create(texton_rows_full, texton_cols_full, CV_8UC1);
        final_result_full.create(texton_rows_full, texton_cols_full, CV_8UC1);
    }
    
    MPI_Gatherv(local_texton.data, local_texton_rows * texton_cols_full, MPI_UNSIGNED_CHAR,
                texton_full.data, t_recv_counts.data(), t_displacements.data(),
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_result.data, local_texton_rows * texton_cols_full, MPI_UNSIGNED_CHAR,
                final_result_full.data, t_recv_counts.data(), t_displacements.data(),
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // --- Phase 6: Finalization ---
    double total_end_time = MPI_Wtime();
    if (rank == 0) {
        //print_matrix(texton_full, "Intermediate Texton Image");
        //print_matrix(final_result_full, "Final LTxXORp Result");
        cout << "Total Runtime: " << (total_end_time - total_start_time) * 1000.0 << " ms" << endl;
    }

    MPI_Finalize();
    return 0;
}
