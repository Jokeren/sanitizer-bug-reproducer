#!/bin/bash

CUDA_PATH=/usr/local/cuda

make clean
make
cd samples/srad
make clean
make
LD_LIBRARY_PATH=../../:/usr/local/cuda-10.1/extras/Sanitizer/:$LD_LIBRARY_PATH ./srad 100 0.5 502 458
cd ../..
