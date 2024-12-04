#include "stdio.h"

__global__ void contrast(float *normalized, float *contrast, int max) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (normalized[row * max + col] > 0) {
    atomicAdd(&contrast[0],
              ((row - col) * (row - col)) * normalized[row * max + col]);
  }
}

__global__ void calculate_IDM(float *norm, float *IDM, int Max) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (norm[row * Max + col] > 0) {
    atomicAdd(&IDM[0],
              norm[row * Max + col] / (1 + ((row - col) * (row - col))));
  }
}

__global__ void calculate_entropy(float *norm, float *entropy, int Max) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (norm[row * Max + col] > 0) {
    atomicAdd(&entropy[0],
              (norm[row * Max + col] * log10f(norm[row * Max + col])));
  }
}
