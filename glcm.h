// glcm.h
#ifndef GLCM_H
#define GLCM_H

void glcm_0_nop(int *matrix, int *glcm, int n_col, int n_row, int glcm_max);
void glcm_optimized(int *matrix, int *glcm, int n_col, int n_row, int glcm_max);

//__global__ void gclm(int *matrix, int *glcm, int n, int c, int max);

#endif // GLCM_H
