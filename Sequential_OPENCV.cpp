#include <stdio.h>
#include <ctime>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double getCurrentTime()
{
    struct timespec currentTime;
    clock_gettime(CLOCK_MONOTONIC, &currentTime);
    return (double)currentTime.tv_sec * 1000.0 + (double)currentTime.tv_nsec / 1000000.0;
}

int t_Xor[3][3] = {0};
int txt_img[2080][2080] = {0};

int wgt[3][3] = {
    {8, 4, 2},
    {16, 0, 1},
    {32, 64, 128}
};

int casecheck(int a, int b, int c, int d) {
   if (a == b && b == c && c == d && d == a)
        return 7;
    else if (a == b)
        return 1;
    else if (b == d)
        return 2;
    else if (c == d)
        return 3;
    else if (a == c)
        return 4;
    else if (a == d)
        return 5;
    else if (c == b)
        return 6;
    
    else
        return 0;
}

int Texton_weight(int fx, int fy, int lx, int ly) {
    int k = 0, l = 0;

    for (int i = fx; i < lx; i++) {
        for (int j = fy; j < ly; j++) {
            //printf("%d - %d\n", txt_img[fx + 1][fy + 1], txt_img[i][j]);
            if (txt_img[fx +1][fy +1] == txt_img[i][j])
                t_Xor[k][l] = 0;
            else {
                t_Xor[k][l] = 1;
            }
            l++;
        }
        l = 0;
        k++;
    }

    int xor_S = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            xor_S = xor_S + (t_Xor[i][j] * wgt[i][j]);
        }
    }
    return xor_S;
}

int main() {
  int i, j;

  Mat image = imread("sample_image/30x30.jpg");
  Mat hsv_image;
  cvtColor(image, hsv_image, COLOR_BGR2HSV);
  int m_r = hsv_image.rows / 2, m_c = hsv_image.cols / 2;
  int img[hsv_image.rows][hsv_image.cols] = {0};
  int Main_res[m_r][m_c] = {0};

  
  for (i = 0; i < hsv_image.rows; i++) {
    for (j = 0; j < hsv_image.cols; j++) {
      Vec3b hsv_pixel = hsv_image.at<Vec3b>(i, j);
      int hue = hsv_pixel[0];
      int saturation = hsv_pixel[1];
      int value = hsv_pixel[2];
      img[i][j] = value;
      printf("%3d ", value);
      }
    printf("\n");
    }

    int fx = 0;
    int fy = 0;
    int lx = hsv_image.rows;
    int ly = hsv_image.cols;

    int k = 0, l = 0;
  double startTime1 = getCurrentTime();
  for (i = fx; i < lx; i = i + 2) {
    for (j = fy; j < ly; j = j + 2) {
      int a, b, c, d;
      a = img[i][j], b = img[i][j + 1], c = img[i + 1][j], d = img[i + 1][j + 1];
      int rs = casecheck(a, b, c, d);
      txt_img[k][l] = rs;
      l++;
      }
      l = 0;
      k++;
    }
  double endTime1 = getCurrentTime();
  double totalTime1 = endTime1 - startTime1;
  
  printf("___________________\n");
  printf("\nTexton image\n \n");

  for (i = fx; i < m_r; i++) {
    for (j = fy; j < m_c; j++)
      printf("%2d  ", txt_img[i][j]);
      printf("\n");
    }

  printf("___________________\n");
  printf("\nTexton Weight image\n \n");

  
  double startTime2 = getCurrentTime();
  for (i = fx; i < m_r; i++) {
    for (j = fy; j < m_c; j++) {
      if (i == 0 || i == m_r - 1 || j == 0 || j == m_c - 1) {
        Main_res[i][j] = txt_img[i][j];
        } 
      else{
        Main_res[i][j] = Texton_weight(i - 1, j - 1, i + 2, j + 2);
          }
      printf("%3d  ", Main_res[i][j]);
      }
    printf("\n");
    }
  double endTime2 = getCurrentTime();
  double totalTime2 = endTime2 - startTime2;

  printf("Elapsed time for texton : %f mili seconds\n", totalTime1);
  printf("Elapsed time for LTxXORp: %f mili seconds\n", totalTime2);
  printf("Elapsed time: %f mili seconds\n", totalTime1+totalTime2);
  return 0;
}
