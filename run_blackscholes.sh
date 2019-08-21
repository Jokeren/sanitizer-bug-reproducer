#!/bin/bash

CUDA_PATH=/usr/local/cuda

cd samples/BlackScholes
make clean
make
LD_LIBRARY_PATH=../../:/usr/local/cuda-10.1/extras/Sanitizer/:$LD_LIBRARY_PATH ./BlackScholes
cd ../..
