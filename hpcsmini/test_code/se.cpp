#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  int my_rank, comm_size;
  int root_rank = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int data_size = 100; // Total number of elements to scatter
  int send_count = data_size / comm_size; // Elements per process

  int send_data[send_count];
  int recv_data[send_count];

  if (my_rank == root_rank) {
    for (int i = 0; i < data_size; i++) {
      send_data[i] = i;
    }
  }

  MPI_Scatter(send_data, send_count, MPI_INT, recv_data, send_count, MPI_INT, root_rank, MPI_COMM_WORLD);

  // Process data received from root process
  for (int i = 0; i < send_count; i++) {
    printf("Process %d received element %d\n", my_rank, recv_data[i]);
  }

  MPI_Finalize();

  return 0;
}
