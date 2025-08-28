#!/bin/bash
CORE=4
IMAGE_DIR=Data/test
RESULT_FILE=performance_results.csv

# Header for CSV
echo "Image,Sequential(ms),MPI(ms),CUDA(ms),MPI Speedup,MPI Efficiency(%),CUDA Speedup,CUDA Efficiency(%)" > $RESULT_FILE

# Loop over all images
for img in $(find $IMAGE_DIR -type f -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg"); do
    echo "Processing $img ..."
    make run-serial IMAGE=$img >> $RESULT_FILE
    make run-mpi CORE=$CORE IMAGE=$img >> $RESULT_FILE
    make run-cuda IMAGE=$img >> $RESULT_FILE
    make run-cuda-shared IMAGE=$img >> $RESULT_FILE
done

echo "Benchmark completed! Results saved in $RESULT_FILE"
