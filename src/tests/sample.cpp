#include <stdio.h>
#include<iostream>
#include <time.h>
#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  int my_rank, num_procs;

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get the rank and number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Check that we have 3 processes
  if (num_procs != 3) {
    printf("This program must be run with 3 processes.\n");
    MPI_Finalize();
    return 1;
  }

  // Allocate each process a different task
  int task_id = my_rank % 6;

  // Perform the task
  switch (task_id) {
    case 0:
      printf("Process %d is performing task 1.\n", my_rank);
      break;
    case 1:
      printf("Process %d is performing task 2.\n", my_rank);
      break;
    case 2:
      printf("Process %d is performing task 3.\n", my_rank);
      break;
    case 3:
      printf("Process %d is performing task 4.\n", my_rank);
      break;
    case 4:
      printf("Process %d is performing task 5.\n", my_rank);
      break;
    case 5:
      printf("Process %d is performing task 6.\n", my_rank);
      break;
    default:
      printf("Process %d has been assigned an invalid task ID.\n", my_rank);
  }

  // Finalize MPI
  MPI_Finalize();

  return 0;
}
