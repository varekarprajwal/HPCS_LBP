#include <iostream>
#include <iomanip>
#include <ctime>
#include <vector>
#include <omp.h> // Include the OpenMP header
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// The weight matrix remains a constant global or can be defined locally.
const int wgt[3][3] = {
    {8, 4, 2},
    {16, 0, 1},
    {32, 64, 128}
};

// --- HELPER FUNCTION TO PRINT A CV::MAT ---
void print_matrix(const Mat& mat_to_print, const string& name) {
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


double getCurrentTime() {
    struct timespec currentTime;
    clock_gettime(CLOCK_MONOTONIC, &currentTime);
    return (double)currentTime.tv_sec * 1000.0 + (double)currentTime.tv_nsec / 1000000.0;
}

// Same casecheck function, but now it returns a uchar.
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
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    Mat image = imread(argv[1]);
    if (image.empty()) {
        cerr << "Error: Couldn't load input image." << endl;
        return -1;
    }

    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_BGR2HSV);

    vector<Mat> hsvChannels;
    split(hsv_image, hsvChannels);
    Mat v_channel = hsvChannels[2]; // This is our source image, type CV_8UC1 (uchar)
    //print_matrix(v_channel, "V channel Image");

    const int m_r = v_channel.rows / 2;
    const int m_c = v_channel.cols / 2;

    // Use cv::Mat with the correct uchar type (CV_8UC1) for all images.
    Mat txt_img(m_r, m_c, CV_8UC1);

    // --- Texton Calculation ---
    double startTime1 = getCurrentTime();
    
    // Parallelize the outer loop. OpenMP will automatically handle
    // distributing the rows (iterations of 'i') among the available threads.
    #pragma omp parallel for
    for (int i = 0; i < m_r; ++i) {
        // Get pointers to the two source rows to avoid repeated .at() calls
        const uchar* p_src1 = v_channel.ptr<uchar>(i * 2);
        const uchar* p_src2 = v_channel.ptr<uchar>(i * 2 + 1);
        // Get pointer to the destination row
        uchar* p_dest = txt_img.ptr<uchar>(i);

        for (int j = 0; j < m_c; ++j) {
            uchar a = p_src1[j * 2];
            uchar b = p_src1[j * 2 + 1];
            uchar c = p_src2[j * 2];
            uchar d = p_src2[j * 2 + 1];
            p_dest[j] = casecheck(a, b, c, d);
        }
    }
    double endTime1 = getCurrentTime();

    // --- LTxXORp Calculation ---
    // Initialize the result image by copying the texton image.
    // This efficiently handles the borders, as they remain unchanged.
    Mat main_res = txt_img.clone();

    double startTime2 = getCurrentTime();
    
    // Parallelize the outer loop for the LTxXORp calculation.
    // Each thread will process a separate chunk of rows from the texton image.
    #pragma omp parallel for
    for (int i = 1; i < m_r - 1; ++i) {
        // Get pointers to the previous, current, and next rows of the texton image
        const uchar* p_prev = txt_img.ptr<uchar>(i - 1);
        const uchar* p_curr = txt_img.ptr<uchar>(i);
        const uchar* p_next = txt_img.ptr<uchar>(i + 1);
        // Get pointer to the destination row in the result matrix
        uchar* p_dest = main_res.ptr<uchar>(i);

        for (int j = 1; j < m_c - 1; ++j) {
            const uchar center_val = p_curr[j];
            
            // Inlined logic from Texton_weight function, removing function call overhead
            // and the global variable. (condition) evaluates to 0 or 1, creating branchless code.
            int xor_S = 0;
            xor_S += (p_prev[j - 1] != center_val) * wgt[0][0];
            xor_S += (p_prev[j]     != center_val) * wgt[0][1];
            xor_S += (p_prev[j + 1] != center_val) * wgt[0][2];
            xor_S += (p_curr[j - 1] != center_val) * wgt[1][0];
            // xor_S += (p_curr[j]  != center_val) * wgt[1][1]; // Center is always 0
            xor_S += (p_curr[j + 1] != center_val) * wgt[1][2];
            xor_S += (p_next[j - 1] != center_val) * wgt[2][0];
            xor_S += (p_next[j]     != center_val) * wgt[2][1];
            xor_S += (p_next[j + 1] != center_val) * wgt[2][2];

            p_dest[j] = saturate_cast<uchar>(xor_S); // Safely cast to uchar
        }
    }
    double endTime2 = getCurrentTime();
    
    // --- PRINT THE FINAL RESULT ---
    //print_matrix(txt_img, "Intermediate Texton Image");
    //print_matrix(main_res, "Final LTxXORp Result");


    double totalTime1 = endTime1 - startTime1;
    double totalTime2 = endTime2 - startTime2;

    //printf("\n____________________________\n\n");
    printf("Elapsed time for texton : %.4f milliseconds\n", totalTime1);
    printf("Elapsed time for LTxXORp: %.4f milliseconds\n", totalTime2);
    printf("Total elapsed time      : %.4f milliseconds\n", totalTime1 + totalTime2);

    return 0;
}