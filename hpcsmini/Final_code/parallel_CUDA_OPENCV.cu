
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
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



__device__ int getGlobalID_3D_3D(){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
  }

__device__ int casecheck(int a, int b, int c, int d) {
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

__global__ void CUDA_START(){
  printf("\nCUDA HARDWARE ACTIVATED\n");
  printf("\n____________________\n");
}
__global__ void calculate_TEXTON_CUDA(int* img_s, int* d_t_img, int rows, int cols)  {
  int x = getGlobalID_3D_3D();
  int a, b, c, d, r, i, q;

  q=x%(cols/2);
  r =x/(cols/2);
  i=((r*2)*cols)+(q*2);
  a = img_s[i], b = img_s[i + 1], c = img_s[cols+i], d = img_s[cols+i+1];

  int rs = casecheck(a, b, c, d);

  d_t_img[x]=rs;
}

__global__ void calculate_LBP_CUDA(int* txt_img, int* Main_res, int rows, const int cols)  {
  int i = getGlobalID_3D_3D();
  //int x =i+(5+1+((i/3)*2));
  int x =i+(cols+1+(i/(cols-2)*2));
  const int a=cols+1,b=cols,c=cols-1,d=1,e=1,f=cols-1,g=cols,h=cols+1;

   if ( i<(rows*cols)-((rows*2)+(cols*2)-4))
   {
    if (txt_img[x] == txt_img[x-a])//6
      txt_img[x-a] = 0;
    else
      txt_img[x-a] = 1;

    if (txt_img[x] == txt_img[x-b]) //5
      txt_img[x-b] = 0;
    else
      txt_img[x-b] = 1;

    if (txt_img[x] == txt_img[x-c]) //4
      txt_img[x-c] = 0;
    else
      txt_img[x-c] = 1;

    if (txt_img[x] == txt_img[x-d]) //1
      txt_img[x-d] = 0;
    else
      txt_img[x-d] = 1;

    if (txt_img[x] == txt_img[x+e])//1
      txt_img[x+e] = 0;
    else
      txt_img[x+e] = 1;

    if (txt_img[x] == txt_img[x+f]) //4
      txt_img[x+f] = 0;
    else
      txt_img[x+f] = 1;

    if (txt_img[x] == txt_img[x+g]) //5
      txt_img[x+g] = 0;
    else
      txt_img[x+g] = 1;

    if (txt_img[x] == txt_img[x+h]) //6
      txt_img[x+h] = 0;
    else
      txt_img[x+h] = 1;

  int xor_S = 0;
  const int wgt[9] = {8, 4, 2,16, 0, 1,32, 64, 128};
  xor_S =(txt_img[x-a] * wgt[0])+(txt_img[x-b] * wgt[1])+(txt_img[x-c] * wgt[2])+(txt_img[x-d] * wgt[3])+(txt_img[x+e] * wgt[5])+(txt_img[x+f] * wgt[6])+(txt_img[x+g] * wgt[7])+(txt_img[x+h] * wgt[8]);
  Main_res[x]=xor_S;
   }
}

int main() {
  CUDA_START<<<1,1>>>();
  int i, j;
  int t_m_r,t_m_c;
  int i_m_r,i_m_c;
  int *h_img_r = new int[2000*2000];
  Mat image = imread("img1.jpg");

  if (image.empty()) {
    cerr << "Error: Couldn't load input image." << endl;
    return -1;
    }
  
  Mat hsv_image;
  cvtColor(image, hsv_image, COLOR_BGR2HSV);
  i_m_r = hsv_image.rows ;
  i_m_c = hsv_image.cols ;
  t_m_r = hsv_image.rows / 2;
  t_m_c = hsv_image.cols / 2;


  int k=0;
  for (i = 0; i < hsv_image.rows; i++) {
    for (j = 0; j < hsv_image.cols; j++) {
      Vec3b hsv_pixel = hsv_image.at<Vec3b>(i, j);
      int value = hsv_pixel[2];
      h_img_r[k]=value;
      printf("%3d ", h_img_r[k]);
      k++;
      }
    printf("\n");
  }
  printf("\n____________________\n");



  printf("\nTexton image\n \n");

  int* d_img;
  int* d_t_img;
  int imageSize = i_m_r * i_m_c * sizeof(int);
  int t_imageSize = t_m_r * t_m_c * sizeof(int);

  cudaMalloc((void **)&d_img, imageSize);
  cudaMalloc((void **)&d_t_img, imageSize);


  cudaMemcpy(d_img, h_img_r, imageSize, cudaMemcpyHostToDevice);


  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks1((i_m_c + threadsPerBlock.x - 1) / threadsPerBlock.x, (i_m_r + threadsPerBlock.y - 1) / threadsPerBlock.y);
  dim3 numBlocks2((t_m_c + threadsPerBlock.x - 1) / threadsPerBlock.x, (t_m_r + threadsPerBlock.y - 1) / threadsPerBlock.y);
  

  double startTime1 = getCurrentTime();

  calculate_TEXTON_CUDA<<<numBlocks1, threadsPerBlock>>>(d_img, d_t_img, i_m_r , i_m_c);

  double endTime1 = getCurrentTime();
  int *h_T_img = new int[t_m_r *t_m_r ];
  cudaMemcpy(h_T_img, d_t_img, t_imageSize, cudaMemcpyDeviceToHost);

  for (i = 0; i < t_m_r; i++) {
    for (j = 0; j < t_m_c; j++) {
      printf("%4d", h_T_img[i*t_m_c+j]);
        }
    printf("\n");
  }

  printf("\n___________________\n");
  printf("\nTexton Weight image\n \n");


  int *h_Txt_img = new int[t_m_r *t_m_r ];
  cudaMalloc((void **)&h_Txt_img, imageSize);

  int *dr_Txt_img = new int[t_m_r *t_m_r ];
  cudaMalloc((void **)&dr_Txt_img, imageSize);

  cudaMemcpy(dr_Txt_img, h_T_img, imageSize, cudaMemcpyHostToDevice);
  cudaMemcpy(h_Txt_img, h_T_img, imageSize, cudaMemcpyHostToDevice);

  double startTime2 = getCurrentTime();

  calculate_LBP_CUDA<<<numBlocks2, threadsPerBlock>>>(dr_Txt_img, h_Txt_img, t_m_r , t_m_c);
  //calculate_LBP_CUDA<<<1,25>>>(dr_Txt_img, h_Txt_img, t_m_r , t_m_c);

  double endTime2 = getCurrentTime();

  int *MM_T_img = new int[t_m_r *t_m_r ];
  cudaMemcpy(MM_T_img, h_Txt_img, t_imageSize, cudaMemcpyDeviceToHost);

  for (i = 0; i < t_m_r; i++) {
    for (j = 0; j < t_m_c; j++) {
      printf("%4d", MM_T_img[i*t_m_c+j]);
        }
    printf("\n");
  }
  double totalTime1 = endTime1 - startTime1;
  printf("Time taken by Parallel to calculate Texton code: %.5f milliseconds\n", totalTime1);
  double totalTime2 = endTime2 - startTime2;
  printf("Time taken by Parallel to calculate LTxXORp code: %.5f milliseconds\n", totalTime2);
  printf("Elapsed time: %f mili seconds\n", totalTime1+totalTime2);

  return 0;
}
