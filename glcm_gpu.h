
// glcm_gpu.h
#ifndef GLCM_GPU_H
#define GLCM_GPU_H

__global__ void glcm_cuda_0(int *matrix, int *glcm, int n_col, int n_row,
                            int glcm_max);
__global__ void glcm_cuda_45(int *matrix, int *glcm, int n_col, int n_row,
                             int glcm_max);

__global__ void glcm_cuda_direction(int *matrix, int *glcm, int n_col,
                                    int n_row, int glcm_max, int dx, int dy);
#endif // GLCM_GPU_H
