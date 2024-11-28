
// glcm_gpu.h
#ifndef GLCM_GPU_H
#define GLCM_GPU_H

__global__ void glcm_cuda_0(int *matrix, int *glcm, int n_col, int n_row,
                            int glcm_max);

// not used deprecated us the cuda_directions
__global__ void glcm_cuda_45(int *matrix, int *glcm, int n_col, int n_row,
                             int glcm_max);

__global__ void glcm_cuda_direction(int *matrix, int *glcm, int n_col,
                                    int n_row, int glcm_max, int dx, int dy);

__global__ void norm(int *glcm, float *glcm_normalized, int max_value, int sum);
#endif // GLCM_GPU_H
