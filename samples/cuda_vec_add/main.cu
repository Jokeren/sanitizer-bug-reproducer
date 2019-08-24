#include <cstdio>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>


static const size_t N = 1000;


void init(int *p, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    p[i] = i;
  }
}


void output(int *p, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("index %zu: %d\n", i, p[i]);
  }
}


__global__
void vecAdd(int *l, int *r, int *p, size_t N, size_t iter) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t i = 0; i < iter; ++i) {
    if (idx < N) {
      p[idx] += l[idx] + r[idx];
    }
  }
}


int main(int argc, char *argv[]) {
  int l[N], r[N], p[N];
  int *dl, *dr, *dp;

  init(l, N);
  init(r, N);

  cudaMalloc(&dl, N * sizeof(int));
  cudaMalloc(&dr, N * sizeof(int));
  cudaMalloc(&dp, N * sizeof(int));

  cudaMemcpy(dl, l, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dr, r, N * sizeof(int), cudaMemcpyHostToDevice);

  size_t threads = 256;
  size_t blocks = (N - 1) / threads + 1;

  vecAdd<<<blocks, threads>>>(dl, dr, dp, N, 100);

  cudaMemcpy(p, dp, N * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dl);
  cudaFree(dr);
  cudaFree(dp);

  cudaStreamSynchronize(NULL);

  return 0;
}
