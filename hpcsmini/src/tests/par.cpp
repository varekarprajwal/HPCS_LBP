#include <stdio.h>
#include<iostream>
#include <time.h>
#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


const int wgt[3][3] = {
    {8, 4, 2},
    {16, 0, 1},
    {32, 64, 128}
};




int casecheck(int a, int b, int c, int d) {
    //printf("%d %d %d %d",a,b,c,d);
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
    
    int i, j ,run;
    int size,rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double start_time, end_time;
    double total_time_taken;
    double start_time1, end_time1;
    double total_time_taken1;
    MPI_Status sta;
    int *img_r = new int[1000*1000];
    int t_m_r,t_m_c,i_m_r,i_m_c;
    int ofset_r=0,ofset_text=0,ofset=1,ofset_ss=0;
    int rank_offset[size]={0};
    
    
    if (rank == 0)
    {
        Mat image = imread("img1.jpg");

        Mat hsv_image;
        cvtColor(image, hsv_image, COLOR_BGR2HSV);
        i_m_r = hsv_image.rows ;
        i_m_c = hsv_image.cols ;
        t_m_r = hsv_image.rows / 2;
        t_m_c = hsv_image.cols / 2;
        
        //int* img_r = new int[hsv_image.rows * hsv_image.cols];
        int k=0;
        for (i = 0; i < hsv_image.rows; i++) {
            for (j = 0; j < hsv_image.cols; j++) {
                Vec3b hsv_pixel = hsv_image.at<Vec3b>(i, j);
                int value = hsv_pixel[2];
                img_r[k]=value;
                printf("%4d", img_r[k]);
                k++;   
            }
           printf("\n");
        }

        //printf("_______________________________________________________________\n");
        /* for (i = 0; i <i_m_c*i_m_r; i++) {    
            printf("%d ", img_r[i]);
            //if( i%i_m_c-1==0)
            //printf("\n");
        }*/
        for (i = 0; i <t_m_r; i++){
        if (i!=0 && i%size==0){
            ofset++;
        }
        //printf("%d ",i);
        MPI_Send(&img_r[(2*i_m_c)*(i)], 2*i_m_c, MPI_INT, i%size, (i%size)+(size*(ofset-1)), MPI_COMM_WORLD);
        //printf("rank %d and start addres %d ofset_tg %d\n", i%size,(2*i_m_c)*(i),(i%size)+(size*(ofset-1)));
      
        rank_offset[i%size]=ofset;

        //printf("ofset value %d\n",ofset);
        //printf("rank %d and start addres %d ofset %d\n", i%size,(2*i_m_c)*(ofset-1),(i%size)+(size*(ofset-1)));
        
        //MPI_Send(&img_r[50], (i_m_r/size)*i_m_c, MPI_INT, 1, 1, MPI_COMM_WORLD);
        }
    

        /*
        for (i = 0; i <t_m_r; i++){
        //printf("%d ", ((i_m_r/size)*i_m_c)*i);
        int s=0;
        MPI_Send(&img_r[(2*i_m_c)*i], 2*i_m_c, MPI_INT, i, i, MPI_COMM_WORLD);
        s++;
        if(s==size-1)
        s=0;
        }*/
    }

    
    MPI_Scatter(&rank_offset, 1, MPI_INT,&ofset_r, 1,MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&i_m_r,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&i_m_c,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&t_m_r,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&t_m_c,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&ofset,1,MPI_INT,0,MPI_COMM_WORLD);
    int img_s[i_m_c*i_m_r];
    int *my_texton =new int[t_m_c*t_m_r];
    int *my_r_texton =new int[t_m_c*t_m_r];
    int Main_res[t_m_r][t_m_c] = {0};
    int t_Xor[3][3] = {0};
    int txt_img[t_m_r][t_m_c] = {0};
    //MPI_Scatter(&img_r, 2*i_m_c, MPI_INT,&img_s, 2*i_m_c,MPI_INT,0, MPI_COMM_WORLD);


    if(rank >=0){

        //printf("recvicer rank %d and ofset %d\n", rank,ofset_r);
        
        for (i = 0; i < ofset_r;i++) {
        //printf("star loction %d ,%d, ofset %d\n", (2*i_m_c)*i,rank,rank+(i*size));
        MPI_Recv(&img_s[(2*i_m_c)*i], 2*i_m_c, MPI_INT, 0, rank+(i*size), MPI_COMM_WORLD, &sta);
        //printf("recvicer add %d rank %d and ofset-tag %d\n", (2*i_m_c)*i,rank,rank+(i*size));
        }
    
    //printf(" from rank %d \n",rank);
    /*for (i = 0; i < 2*i_m_c*(ofset_r); i++){
    printf("%d form rank %d coint %d\n", img_s[i],rank,i);
    printf("%d(%d) ",rank,img_s[i]);
    }*/
    start_time = MPI_Wtime();
    int k=0,r=0;
    
    for (i = 0; i < (i_m_c*ofset_r);) {
        //printf("%d ", i_m_r);
        int a, b, c, d,skip;
        a = img_s[i], b = img_s[i + 1], c = img_s[i_m_c+i], d = img_s[i_m_c+i+1];
        int rs = casecheck(a, b, c, d);
        //printf("vale  %d %d %d %d\n",a,b,c,d);
        //printf("pos  %d %d %d %d\n", i,i+1,i_m_c+i,i_m_c+i+1);

        
        my_texton[r]=rs;
        //printf("%d at vale and pos %d    rank %d\n", my_texton[r],r, rank);
        //printf("%d from rs and rank %d\n", rs,rank);

        r++;
       //printf("%d from %d", rs,rank);
        
            k+=2;
            if (k%i_m_c==0){
                //printf("%d \n",i);
                i+=i_m_c+2;
            }
            else{
                i+=2;
            }
            //printf("\n");
        }

        //MPI_Send(&my_texton, t_m_c*ofset_r, MPI_INT, rank, rank+10, MPI_COMM_WORLD);
        //printf("%d addrs stat %d offset tag of rank %d  \n",t_m_c*ofset_r,rank+10,rank);
    end_time = MPI_Wtime();
    double total_time_taken = end_time - start_time;

    printf("Total time taken: %f seconds\n", total_time_taken);
       
    }
    
    //printf(" size %d  ", rank*t_m_c*ofset_r);
    MPI_Gather(&my_texton[0], t_m_c*ofset_r, MPI_INT,&my_r_texton[rank*t_m_c*ofset_r], t_m_c*ofset_r,MPI_INT,0, MPI_COMM_WORLD);
    //int x[20]={0,1,2,3,4,5,6,7,8,9,00,11,22,33,44,55,66,77,88,99};
    //MPI_Gather(&x[rank],2, MPI_INT,&my_r_texton[rank+1], 2,MPI_INT,0, MPI_COMM_WORLD); 
    
    
    /*if (rank == 0)
    {   
        for(i = 0; i <size; i++){
        //MPI_Recv(&my_r_texton, t_m_c*ofset_r, MPI_INT, i%size, i+10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf(" recive %d addrs stat %d offset tag of rank %d  \n",t_m_c*i,i+10,i%size);
        }
        for (i = 0; i <t_m_c*t_m_r; i++)
        printf("%d  ", my_r_texton[i]);
    }*/

    
    if (rank==0){
        /*for (i = 0; i <t_m_c*t_m_r; i++){
        printf("%d ", my_r_texton[i]);
        }*/
    printf("___________________\n");
    printf("\nTexton image\n \n");

    /*for (i = 0; i <size; i++){
        printf("ofset value %d\n",rank_offset[i]);
    }*/
    
    printf("The swapped matrix is: \n");

    for (i = 0; i < t_m_r; i++) {
        for (j = 0; j < t_m_c; j++) {
            txt_img[i][j]=my_r_texton[i*t_m_c+j];
           printf("%d ", txt_img[i][j]);
        }
        printf("\n");
    }
    





    printf("___________________\n");
    printf("\nTexton Weight image\n \n");

    for (i = 0; i <t_m_r-2; i++){
        if (i!=0 && i%size==0){
            ofset_ss++;
        }
        //MPI_Send(&my_r_texton[(t_m_c)*(i)], 3*t_m_c, MPI_INT, i%size, (i%size)+(size*20), MPI_COMM_WORLD);
        //printf("rank %d and start addres %d ofset_tg %d\n", i%size,(t_m_c)*(i),(i%size)+(size*20));
        }
       // printf("\n");

  start_time1 = MPI_Wtime();
        for (i = 0; i < t_m_r; i++) {
            for (j = 0; j < t_m_c; j++) {
            //txt_img[i][j]=my_r_texton[i*t_m_c+j];

            //printf("%d  ", txt_img[i][j]);

            if (i == 0 || i == t_m_r - 1 || j == 0 || j == t_m_c - 1) {
                Main_res[i][j] = txt_img[i][j];
            } 
            else 
                //Main_res[i][j] = Texton_weight(i - 1, j - 1, i + 2, j + 2);
            {
            int k = 0, l = 0;
            int fx=i - 1,fy= j - 1, lx=i + 2,ly= j + 2;


            for (int i = fx; i < lx; i++) {
                for (int j = fy; j < ly; j++) {
            
                //printf("%d - %d\n", txt_img[fx + 1][fy +1], txt_img[i][j]);
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
            Main_res[i][j] = xor_S;

            }
            printf("%4d", Main_res[i][j]);
        }
        printf("\n");
    }
    end_time1 = MPI_Wtime();
    total_time_taken1 = (end_time1 - start_time1)*1000;

    }
if(rank==0)
{ /*for (i = 0; i <t_m_r;i++ ) {
        for (j = 0; j < t_m_c; j++) {
        
        printf("%4d", my_g_texton[i*t_m_c+j]);

}
printf("\n");
}*/
    printf("Elapsed time for texton : %f mili seconds\n", total_time_taken);
    printf("Elapsed time for LTxXORp: %f mili seconds\n", total_time_taken1);
    printf("Elapsed time: %f mili seconds\n", total_time_taken+total_time_taken1);
}



    
    MPI_Finalize();
    return 0;
}
