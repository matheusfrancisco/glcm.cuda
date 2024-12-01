#include "stdio.h"

__global__ void contrast(float *normalized, float *contrast, int max) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (normalized[row * max + col] > 0) {
    atomicAdd(&contrast[0],
              ((row - col) * (row - col)) * normalized[row * max + col]);
  }
}
