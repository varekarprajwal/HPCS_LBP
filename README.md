HPCS_LBP

##CREATING THE REQUIRED ENVIROMENT

## Required
!sudo apt update

!sudo apt install libopencv-dev python3-opencv

#May be Required
!apt-get install -y mpich

!apt-get install -y openmpi-bin openmpi-doc openmpi-dev

!apt-get install -y g++

##Run command
#COMMAND RUN SEQUENTIAL CODE
!g++ Sequential_OPENCV.cpp -o app `pkg-config --cflags --libs opencv4` && ./app

#COMMAND RUN PARALLEL MPI CODE
!mpic++ Parallel_MPI_OPENCV.cpp -o app `pkg-config --cflags --libs opencv4` && mpirun --allow-run-as-root -np 5 ./app

#COMMAND RUN PARALLEL CUDA CODE
!nvcc -o parallel parallel_CUDA_OPENCV.cu `pkg-config --cflags --libs opencv4` && ./parallel
