#include "glcm.h"
#include <vector>

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

void glcm_0(int *matrix, int *glcm, int n_col, int n_row, int glcm_max) {
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

// Function to get the 8 directions as pairs of (row_offset, col_offset)
std::vector<std::pair<int, int>> get_directions() {
  return {
      {0, 1},   // Right (0°)
      {1, 1},   // Bottom-right (45°)
      {1, 0},   // Bottom (90°)
      {1, -1},  // Bottom-left (135°)
      {0, -1},  // Left (180°)
      {-1, -1}, // Top-left (225°)
      {-1, 0},  // Top (270°)
      {-1, 1}   // Top-right (315°)
  };
}

void glcm_directions(int *matrix, int *glcm, int n_col, int n_row, int glcm_max,
                     int dx, int dy) {
  for (int row = 0; row < n_row; row++) {
    int new_row = row + dx;
    for (int col = 0; col < n_col; col++) {
      int new_col = col + dy;

      int current = matrix[row * n_col + col];

      // Check if the neighboring pixel is within bounds
      if (new_row >= 0 && new_row < n_row && new_col >= 0 && new_col < n_col) {
        int next = matrix[new_row * n_col + new_col];

        // Validate gray levels to prevent out-of-bounds access
        if (current >= 0 && current < glcm_max && next >= 0 &&
            next < glcm_max) {
          glcm[current * glcm_max + next]++;
        }
      }
    }
  }
}

void norm_cpu(const int *glcm, float *glcm_normalized, int max_value, int sum) {
  int size = (max_value + 1) * (max_value + 1);

  for (int idx = 0; idx < size; ++idx) {
    glcm_normalized[idx] = float(glcm[idx]) / float(sum);
  }
}
