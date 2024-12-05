#include "glcm_gpu.h"
#include "stdio.h"

__global__ void glcm_calculation_nol(int *matrix, int *glcm, const int n_row,
                                     const int n_col, int max_value) {

  unsigned int idx = blockIdx.x * n_row + threadIdx.x;
  int i;
  int k = 0;
  for (i = 0; i < n_row; i++) {
    if (idx >= i * n_row && idx < ((i + 1) * n_row) - 1) {
      k = max_value * matrix[idx] + matrix[idx + 1];
      atomicAdd(&glcm[k], 1);
    }
  }
}

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
  // Calculate the linear index for the current pixel
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int total_pixels = n_row * n_col;

  if (idx < total_pixels) {
    // Determine the row and column from idx
    int row = idx / n_col;
    int col = idx % n_col;

    // Calculate neighbor's row and column based on direction
    int neighbor_row = row + dy;
    int neighbor_col = col + dx;

    // Check if the neighbor is within matrix bounds
    if (neighbor_row >= 0 && neighbor_row < n_row && neighbor_col >= 0 &&
        neighbor_col < n_col) {
      // Linear index for neighbor
      int neighbor_idx = neighbor_row * n_col + neighbor_col;

      // Retrieve gray levels for the current pixel and its neighbor
      int current = matrix[idx];
      int neighbor = matrix[neighbor_idx];

      // Validate gray levels to avoid out-of-bounds GLCM updates
      if (current >= 0 && current < glcm_max && neighbor >= 0 &&
          neighbor < glcm_max) {
        // Compute the GLCM index and update the GLCM matrix atomically
        int k = current * glcm_max + neighbor;
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

  if (idx < (max_value + 1) * (max_value + 1) && glcm[idx] != 0) {
    // if (float(glcm[idx]) / float(sum) > 0) {
    //   printf("%f", float(glcm[idx]) / float(sum));
    // }
    glcm_normalized[idx] = float(glcm[idx]) / float(sum);
  }
}

__global__ void transposed(int *transposed, int *glcm, int Max) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  transposed[row * Max + col] = glcm[row * Max + col] + glcm[col * Max + row];
}

// trying to make it by blocks
// nxn block and grid

__global__ void glcm_0_degree(int *matrix, int *glcm, int nx, int ny, int Max) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  int Index = iy * nx + ix;
  int posisi = 0;

  // We iterate over the rows in steps of 2
  for (int i = 0; i < nx; i += 2) {
    // Check if this thread's Index corresponds to a pixel in row i of the image
    // and ensures we don't go out of horizontal bounds for pairs.
    // Condition: Index in [i*nx, (i+1)*nx - 1]
    if (Index >= i * nx && Index < ((i + 1) * nx) - 1) {

      // Horizontal pair: (matrix[Index], matrix[Index + 1])
      posisi = matrix[Index] * Max + matrix[Index + 1];
      atomicAdd(&glcm[posisi], 1);

      // Vertical pair: (matrix[Index + nx], matrix[Index + nx + 1])
      // This uses the corresponding pixel in the next row and the pixel to the
      // right of it.
      posisi = matrix[Index + nx] * Max + matrix[Index + (nx + 1)];
      atomicAdd(&glcm[posisi], 1);
    }
  }
}
