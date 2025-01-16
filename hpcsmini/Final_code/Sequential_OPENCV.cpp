#include <stdio.h>
#include <ctime>
#include <vector>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int t_Xor[3][3] = {0};
int wgt[3][3] = {
    {8, 4, 2},
    {16, 0, 1},
    {32, 64, 128}};

template <typename T>
std::vector<std::vector<T>> Create_image(int r, int c)
{
  // Create a 2D vector with r rows and c columns
  return std::vector<std::vector<T>>(r, std::vector<T>(c));
}
template <typename T>
void Print_image(int &r, int &c, std::vector<std::vector<T>> &temp)
{
  for (int i = 0; i < r; i++)
  {
    for (int j = 0; j < c; j++)
      cout << setw(3) << temp[i][j] << " ";
    printf("\n");
  }
}

double getCurrentTime()
{
  struct timespec currentTime;
  clock_gettime(CLOCK_MONOTONIC, &currentTime);
  return (double)currentTime.tv_sec * 1000.0 + (double)currentTime.tv_nsec / 1000000.0;
}

int casecheck(int a, int b, int c, int d)
{
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

int Texton_weight(int fx, int fy, int lx, int ly, std::vector<std::vector<int>> &txt_img)
{
  int k = 0, l = 0;

  for (int i = fx; i < lx; i++)
  {
    for (int j = fy; j < ly; j++)
    {
      if (txt_img[fx + 1][fy + 1] == txt_img[i][j])
        t_Xor[k][l] = 0;
      else
      {
        t_Xor[k][l] = 1;
      }
      l++;
    }
    l = 0;
    k++;
  }

  int xor_S = 0;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      xor_S = xor_S + (t_Xor[i][j] * wgt[i][j]);
    }
  }
  return xor_S;
}

int main(int argc, char *argv[])
{
  int i, j;

  Mat image = imread(argv[1]);
  if (image.empty())
  {
    cerr << "Error: Couldn't load input image." << endl;
    return -1;
  }
  Mat hsv_image;
  cvtColor(image, hsv_image, COLOR_BGR2HSV);

  vector<Mat> hsvChannels;
  split(hsv_image, hsvChannels);
  Mat img = hsvChannels[2];

  int r = hsv_image.rows, c = hsv_image.cols;
  int m_r = hsv_image.rows / 2, m_c = hsv_image.cols / 2;

  vector<vector<int>> txt_img = Create_image<int>(m_r, m_c);
  vector<vector<int>> Main_res = Create_image<int>(m_r, m_c);


  int k = 0, l = 0;
  double startTime1 = getCurrentTime();
  for (i = 0; i < r; i += 2)
  {
    for (j = 0; j < c; j += 2)
    {
      int a = img.at<uint8_t>(i, j);
      int b = img.at<uint8_t>(i, j + 1);
      int c = img.at<uint8_t>(i + 1, j);
      int d = img.at<uint8_t>(i + 1, j + 1);
      int rs = casecheck(a, b, c, d);
      txt_img[k][l] = rs;
      l++;
    }
    l = 0;
    k++;
  }

  double endTime1 = getCurrentTime();
  double totalTime1 = endTime1 - startTime1;

  img.release();

  double startTime2 = getCurrentTime();
  for (i = 0; i < m_r; i++)
  {
    for (j = 0; j < m_c; j++)
    {
      if (i == 0 || i == m_r - 1 || j == 0 || j == m_c - 1)
      {
        Main_res[i][j] = txt_img[i][j];
      }
      else
      {
        Main_res[i][j] = Texton_weight(i - 1, j - 1, i + 2, j + 2, txt_img);
      }
    }
  }
  //Print_image(m_r, m_c, Main_res);

  double endTime2 = getCurrentTime();
  double totalTime2 = endTime2 - startTime2;
  printf("____________________________\n\n");

  printf("Elapsed time for texton : %f mili seconds\n", totalTime1);
  printf("Elapsed time for LTxXORp: %f mili seconds\n", totalTime2);
  printf("Elapsed time: %f mili seconds\n", totalTime1 + totalTime2);

  return 0;
}
