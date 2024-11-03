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
