#include <cmath>
#include <iostream>
#include <vector>

void calculate_contrast(const std::vector<float> &normalized, float &contrast,
                        int max) {
  contrast = 0.0f;

  for (int row = 0; row < max; ++row) {
    for (int col = 0; col < max; ++col) {
      float value = normalized[row * max + col];
      if (value > 0) {
        float diff = row - col;
        contrast += diff * diff * value;
      }
    }
  }
}
