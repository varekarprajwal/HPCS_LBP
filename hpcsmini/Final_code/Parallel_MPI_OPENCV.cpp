#include <stdio.h>
#include <iostream>
#include <time.h>
#include <mpi.h>
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
int main( int argc, char* argv[]) {

    MPI_Init(&argc, &argv);
    int size,rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status sta;
    
    int i, j ,root_rank=0;
    int t_m_r,t_m_c,i_m_r,i_m_c;

    int rank_offset[size]={0},s_rank_offset[size]={0},rank_offset1[size]={0},text_offset[size]={0};
    int *img_r = new int[2000*2000];

    int counts[size]={0},displacements[size]={0};
    int temp_displacements=0,ofset=1,ofset_r=0;
    int t_counts[size]={0},t_displacements[size]={0},tt_displacements[size]={0},et_counts[size]={0},et_displacements[size]={0};
    
    printf("\nGPU HARDWARE ACTIVATED %d of %d\n",rank,size);

    
    if (rank == 0){
        Mat image = imread("image_sample/1000x1000.png");
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
                img_r[k]=value;
                //printf("%3d ", img_r[k]);
                k++;   
            }
        //printf("\n");
        }

    }
    // SENDING THE DATA
    MPI_Bcast(&i_m_r,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&i_m_c,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&t_m_r,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&t_m_c,1,MPI_INT,0,MPI_COMM_WORLD);

    int *Final =new int[(t_m_c-2) * (t_m_r-2)];
    Final[(t_m_c-2) * (t_m_r-2)] = {0};
    int img_s[i_m_c*i_m_r];
    int txt_simg[t_m_r][t_m_c] = {0};
    int *my_texton =new int[t_m_c*t_m_r];
    int *my_r_texton =new int[t_m_c*t_m_r];
    int *txt_img =new int[t_m_c*t_m_r];
    txt_img[t_m_c*t_m_r]={0};
    int *Main_res =new int[t_m_c*t_m_r];
    Main_res[t_m_c*t_m_r]={0};


    for (i=0;i<t_m_c;i++){
        if (i!=0 && i%size==0)
            ofset++;
        rank_offset[i%size]=ofset;
        //printf("%d form rank %d coint %d\n", rank_offset[i],rank,i);

    }
    
    for (i=0;i<size;i++){
        counts[i]=(2*i_m_c)*rank_offset[i];
        displacements[i]=temp_displacements;
        temp_displacements+=(2*i_m_c)*rank_offset[i];
        //printf("vale  rank %d T-count %d T-diaplacemt %d Rank offset%d \n",rank ,counts[i],displacements[i],rank_offset[i]);
    }
    temp_displacements = 0;
    for (i=0;i<size;i++){
        t_counts[i]=(t_m_c)*rank_offset[i];
        t_displacements[i]=temp_displacements;
        temp_displacements+=(t_m_c)*rank_offset[i];
        //printf("vale  rank %d T-count %d T-diaplacemt %d Rank offset%d \n",rank ,t_counts[i],t_displacements[i],rank_offset[i]);
    }
    
    
    MPI_Scatter(&rank_offset, 1, MPI_INT,&ofset_r, 1,MPI_INT,0, MPI_COMM_WORLD);
    MPI_Scatterv(img_r, counts, displacements, MPI_INT, &img_s,counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    /*for (i = 0; i < 2*i_m_c*ofset_r; i++)
        printf("%d form rank %d coint %d\n",img_s[i],rank,i);*/
    int k=0,r=0,skip=0;
    double startTime1 = getCurrentTime();
        for (i = 0; i < (2*i_m_c*ofset_r);) {
            int a, b, c, d;
            a = img_s[i], b = img_s[i + 1], c = img_s[i_m_c+i], d = img_s[i_m_c+i+1];
            int rs = casecheck(a, b, c, d);
            my_texton[r]=rs;
            //printf("%d at vale and pos %d    rank %d\n", my_texton[r],r, rank);
            r++;
            k+=2;
            if (k%i_m_c==0)
                i+=i_m_c+2;
            else
                i+=2;

            skip++;
        } 
        double endTime1 = getCurrentTime();
        //printf("vale  rank %d T-count %d T-diaplacemt %d Rank offset%d \n",rank ,t_counts[rank],t_displacements[rank],rank_offset[rank]);
        MPI_Gatherv(my_texton, t_counts[rank], MPI_INT, my_r_texton, t_counts, t_displacements, MPI_INT, root_rank, MPI_COMM_WORLD);
        
    if (rank == 0){

    //printf("___________________\n");
    //printf("\nTexton image\n \n");

        for (i = 0; i < t_m_r; i++) {
            for (j = 0; j < t_m_c; j++) {
                txt_simg[i][j]=my_r_texton[i*t_m_c+j];
                //printf("%2d  ", txt_simg[i][j]);
        
            }
            //printf("\n");
        }
    }

    ofset=0;
    for (i=0;i<t_m_r-2;i++){
        if (i!=0 && i%size==0)
            ofset++;
        text_offset[i%size]=ofset;
        //printf("%d form rank %d coint\n", text_offset[i%size],rank);
    }

    int tt_counts[size]={0};
    for (i=0;i<size;i++){
        tt_counts[i]=t_m_r*(3+text_offset[i]);
    }

    temp_displacements = 0;
    for (i=0;i<size;i++){
        counts[i]=(t_m_c)*text_offset[i];
        tt_displacements[i]= temp_displacements;
        temp_displacements += t_m_c + counts[i];   
        //printf("vale  rank %d size count %d T-count %d T-diaplacemt %d Rank offset%d \n",rank ,tt_counts[i],counts[i],tt_displacements[i],text_offset[i]);
    }


    
    MPI_Bcast(&text_offset[size],size,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatterv(txt_simg, tt_counts, tt_displacements, MPI_INT, txt_img, tt_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);
    //printf("Txt vale  rank %d T-count %d T-diaplacemt %d Rank offset%d \n",rank ,tt_counts[rank],tt_displacements[rank],text_offset[rank]);

    
    double startTime2 = getCurrentTime();
        
    
    k=0;
    for (j=0;j<=text_offset[rank];j++){
        int cols=t_m_c;
        int x=((j+1)*cols)+1;

        for (i=0;i<t_m_c-2;i++){
            x=x+i;
            //printf("x=%d \n", x);
            int txt_imgs[9] = {0};
            const int a=cols+1,b=cols,c=cols-1,d=1,e=1,f=cols-1,g=cols,h=cols+1;
            
            if (txt_img[x] == txt_img[x-a])//6
                txt_imgs[1] = 0;
            else
                txt_imgs[1] = 1;

            if (txt_img[x] == txt_img[x-b]) //5
                txt_imgs[2] = 0;
            else
                txt_imgs[2] = 1;

            if (txt_img[x] == txt_img[x-c]) //4
                txt_imgs[3] = 0;
            else
                txt_imgs[3] = 1;

            if (txt_img[x] == txt_img[x-d]) //1
                txt_imgs[4] = 0;
            else
                txt_imgs[4] = 1;

            if (txt_img[x] == txt_img[x+e])//1
                txt_imgs[5] = 0;
            else
                txt_imgs[5] = 1;

            if (txt_img[x] == txt_img[x+f]) //4
                txt_imgs[6] = 0;
            else
                txt_imgs[6] = 1;

            if (txt_img[x] == txt_img[x+g]) //5
                txt_imgs[7] = 0;
            else
                txt_imgs[7] = 1;

            if (txt_img[x] == txt_img[x+h]) //6
                txt_imgs[8] = 0;
            else
                txt_imgs[8] = 1;
            x=((j+1)*cols)+1;
            int xor_S = 0;
            const int wgt[9] = {8, 4, 2,16, 0, 1,32, 64, 128};
            xor_S =(txt_imgs[1] * wgt[0])+(txt_imgs[2] * wgt[1])+(txt_imgs[3] * wgt[2])+(txt_imgs[4] * wgt[3])+(txt_imgs[5] * wgt[5])+(txt_imgs[6] * wgt[6])+(txt_imgs[7] * wgt[7])+(txt_imgs[8] * wgt[8]);
            Main_res[k]=xor_S;
            //printf("%2d  ", Main_res[k]);
            k++;
        }
        //printf("\n");
    }
    
    double endTime2 = getCurrentTime();
 
    temp_displacements = 0;
    for (i=0;i<size;i++){
        et_counts[i]=(t_m_c-2)*(text_offset[i] + 1 );
        et_displacements[i]=temp_displacements;
        temp_displacements += (t_m_c-2)*(1+text_offset[i]);     
        //printf("vale  rank %d T-count %d T-diaplacemt %d Rank offset%d \n",rank ,et_counts[i],et_displacements[i],text_offset[i]);
    }

    MPI_Gatherv(Main_res, et_counts[rank], MPI_INT, Final, et_counts, et_displacements, MPI_INT, root_rank, MPI_COMM_WORLD);
   
    /*
    if (rank == 0){
        printf("\n___________________\n");
        printf("\nTexton Weight image\n \n");
        for (i = 0; i < t_m_r-2; i++) {
            for (j = 0; j < t_m_c-2; j++) {
                printf("%2d  ", Final[i*t_m_c+j]);
            }
            printf("\n");
        }
    }
    */

    double totalTime1 = endTime1 - startTime1;
    printf("Time taken by %d Rank Parallel to calculate Texton code: %.5f milliseconds\n",rank, totalTime1);
    double totalTime2 = endTime2 - startTime2;
    printf("Time taken by %d Rank Parallel to calculate LTxXORp code: %.5f milliseconds\n",rank, totalTime2);
    printf("Elapsed time: %f mili seconds\n", totalTime1+totalTime2);

    MPI_Finalize();
    return 0;
}