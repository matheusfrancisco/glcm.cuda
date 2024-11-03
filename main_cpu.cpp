#include "file.h"
#include "glcm.h"
#include "image.h"
#include <filesystem>
#include <iostream>
#include <unordered_map>

using namespace std;
namespace fs = std::filesystem;

void apply_glcm_0(std::string *file) {
  png_image image_png;
  std::cout << file->c_str() << std::endl;

  open_image_value_32b_array(file->c_str(), &image_png);
  int *matrix =
      (int *)malloc(sizeof(int) * (image_png.width * image_png.height));

  // get the maximum valur of the image
  int max = 0; 
  for (int i = 0; i < (image_png.height * image_png.width); ++i) {
    matrix[i] = image_png.image[i];
    if (matrix[i] > max) {
      max = matrix[i];
    }
  }
  max += 1;
  int *glcm = (int *)malloc(sizeof(int) * (max * max));

  glcm_0_nop(matrix, glcm, image_png.height, image_png.width, max);
}

int main() {
  std::string folder = "data";

  std::unordered_map<fs::path, fs::path, PathHash> file_map =
      get_images(folder);
  for (const auto &file : file_map) {

    // std::cout << image_png.height << std::endl;
    // std::cout << image_png.width << std::endl;
    // for (int i = 0; i < image_png.height * image_png.width; i++) {
    //  std::cout << (int)image_png.image[i] << " ";
    //  std::cout << std::endl;
    //}
  }

  return 0;
}
