#!/bin/bash

CUDA_PATH=/usr/local/cuda

make clean
make INSTRUMENTOR=1
cd samples/cuda_vec_add
make clean
make
LD_LIBRARY_PATH=../../:/usr/local/cuda-10.1/extras/Sanitizer/:$LD_LIBRARY_PATH ./main
cd ../..
