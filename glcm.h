// glcm.h
#include <vector>
#ifndef GLCM_H
#define GLCM_H

void glcm_0_nop(int *matrix, int *glcm, int n_col, int n_row, int glcm_max);
void glcm_0(int *matrix, int *glcm, int n_col, int n_row, int glcm_max);

std::vector<std::pair<int, int>> get_directions();
void glcm_directions(int *matrix, int *glcm, int n_col, int n_row, int glcm_max,
                     int dx, int dy);
//__global__ void gclm(int *matrix, int *glcm, int n, int c, int max);

void norm_cpu(const int *glcm, float *glcm_normalized, int max_value, int sum);
#endif // GLCM_H
