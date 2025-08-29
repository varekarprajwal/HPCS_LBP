#include<stdio.h>
#include<stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
 
    // Get number of processes and check that 4 processes are used
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 4)
    {
        printf("This application is meant to be run with 4 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 
    // Determine root's rank
    int root_rank = 0;
 
    // Get my rank
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
 
    // Define my value
    int my_value;
 
    if(my_rank == root_rank)
    {
        int buffer[4] = {0, 100, 200, 300};
        printf("Values to scatter from process %d: %d, %d, %d, %d.\n", my_rank, buffer[0], buffer[1], buffer[2], buffer[3]);
        MPI_Scatter(buffer, 1, MPI_INT, &my_value, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatter(NULL, 1, MPI_INT, &my_value, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    }
 
    if(my_rank == root_rank)
    printf("Process %d received value = %d.\n", my_rank, my_value);
    printf("Process %d received value = %d.\n", my_rank,size);
 
    MPI_Finalize();
 
    return EXIT_SUCCESS;
}