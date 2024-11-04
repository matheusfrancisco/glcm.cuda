#include "glcm.h"

/* Calculate the glcm for the image array
 * #TODO the function is not optimized
 * Complexity: O(n^4)
 * nop means not optimized
 * */
void glcm_0_nop(int *matrix, int *glcm, int n_col, int n_row, int glcm_max) {
  int row, col, k, l;
  for (row = 0; row < n_row; row++) {
    for (col = 0; col < n_col - 1; col++) {
      for (k = 0; k < glcm_max; k++) {
        for (l = 0; l < glcm_max; l++) {
          if (matrix[row * n_col + col] == k &&
              matrix[row * n_col + col + 1] == l) {
            glcm[k * glcm_max + l]++;
          }
        }
      }
    }
  }
}

void glcm_optimized(int *matrix, int *glcm, int n_col, int n_row, int glcm_max) {
    int row, col;
    for (row = 0; row < n_row; row++) {
        for (col = 0; col < n_col - 1; col++) {
            int current = matrix[row * n_col + col];
            int next = matrix[row * n_col + col + 1];
            // Validate gray levels to prevent out-of-bounds access
            if (current >= 0 && current < glcm_max && next >= 0 && next < glcm_max) {
                glcm[current * glcm_max + next]++;
            }
        }
    }
}
