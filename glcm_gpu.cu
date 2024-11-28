#include "glcm_gpu.h"
#include "stdio.h"

__global__ void glcm_cuda_0(int *matrix, int *glcm, int n_col, int n_row,
                            int glcm_max) {
  // Calculate the total number of possible pairs: (n_row) * (n_col -1)
  unsigned int total_pairs = n_row * (n_col - 1);
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total_pairs) {
    // Determine the row and column from idx
    int row = idx / (n_col - 1);
    int col = idx % (n_col - 1);

    // Calculate the linear indices for the pair
    int current_idx = row * n_col + col;
    int next_idx = current_idx + 1;

    // Retrieve gray levels
    int current = matrix[current_idx];
    int next = matrix[next_idx];

    // Validate gray levels
    if (current >= 0 && current < glcm_max && next >= 0 && next < glcm_max) {
      // Compute the GLCM index
      int k = current * glcm_max + next;

      // Atomically increment the GLCM count
      atomicAdd(&glcm[k], 1);
    }
  }
}

__global__ void glcm_cuda_45(int *matrix, int *glcm, int n_col, int n_row,
                             int glcm_max) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x; // Column index
  int iy = threadIdx.y + blockIdx.y * blockDim.y; // Row index

  if (ix >= n_col - 1 || iy <= 0)
    return; // Out of bounds for 45-degree neighbor

  int idx = iy * n_col + ix;
  int neighbor_idx = (iy - 1) * n_col + (ix + 1);

  int intensity_current = matrix[idx];
  int intensity_neighbor = matrix[neighbor_idx];

  int k = intensity_current * glcm_max + intensity_neighbor;

  atomicAdd(&glcm[k], 1);
}

__global__ void glcm_cuda_direction(int *matrix, int *glcm, int n_col,
                                    int n_row, int glcm_max, int dx, int dy) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int total_pixels = n_row * n_col;

  if (idx < total_pixels) {
    // Determine the row and column from idx
    int row = idx / n_col;
    int col = idx % n_col;

    // Calculate neighbor's row and column
    int neighbor_row = row + dy;
    int neighbor_col = col + dx;

    // Check if neighbor is within bounds
    if (neighbor_row >= 0 && neighbor_row < n_row && neighbor_col >= 0 &&
        neighbor_col < n_col) {
      // Calculate the linear indices for the current pixel and neighbor
      int current_idx = row * n_col + col;
      int neighbor_idx = neighbor_row * n_col + neighbor_col;

      // Retrieve gray levels
      int current = matrix[current_idx];
      int neighbor = matrix[neighbor_idx];

      // Validate gray levels
      if (current >= 0 && current < glcm_max && neighbor >= 0 &&
          neighbor < glcm_max) {
        // Compute the GLCM index
        int k = current * glcm_max + neighbor;

        // Atomically increment the GLCM count
        atomicAdd(&glcm[k], 1);
      }
    }
  }
}

__global__ void norm(int *glcm, float *glcm_normalized, int max_value,
                     int sum) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * max_value + ix;
  __syncthreads();

  if (idx < (max_value + 1) * (max_value + 1)) {
    // if (float(glcm[idx]) / float(sum) > 0) {
    //   printf("%f", float(glcm[idx]) / float(sum));
    // }
    glcm_normalized[idx] = float(glcm[idx]) / float(sum);
  }
}
