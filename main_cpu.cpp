#include "file.h"
#include "glcm.h"
#include "image.h"
#include <cstring>
#include <filesystem>
#include <iostream>
#include <unordered_map>

using namespace std;
namespace fs = std::filesystem;

void apply_glcm_0(std::string *file, bool write_output = false) {

  png_image image_png;
  std::cout << file->c_str() << std::endl;

  // open the image png and put it into an array
  open_image_value_32b_array(file->c_str(), &image_png);

  size_t m_size = (image_png.width * image_png.height) * sizeof(int);
  int *matrix = (int *)malloc(m_size);
  // get the maximum valur of the image
  int max = 0;
  for (int i = 0; i < (image_png.height * image_png.width); ++i) {
    matrix[i] = image_png.image[i];
    if (matrix[i] > max) {
      max = matrix[i];
    }
  }

  max += 2;

  int n_row = image_png.height;
  int n_col = image_png.width;
  std::cout << "max: " << max << std::endl;
  std::cout << "n_row: " << n_row << std::endl;
  std::cout << "n_col: " << n_col << std::endl;

  int glcm_size = (max * max) * sizeof(int);
  int *h_glcm_cpu = (int *)malloc(glcm_size);

  memset(h_glcm_cpu, 0, glcm_size);
  glcm_0(matrix, h_glcm_cpu, n_col, n_row, max);

  if (write_output) {
    std::string r;
    {
      fs::path file_path(file->c_str());
      fs::path new_file_name =
          "result/" + file_path.stem().string() + "_result.txt";
      fs::path new_file_path = file_path.parent_path() / new_file_name;
      r = new_file_path.string();
    }

    write_image_matrix(r, h_glcm_cpu, max, max);
  }
  free(matrix);
  free(h_glcm_cpu);
}

int main() {
  std::string folder = "data";

  std::unordered_map<fs::path, fs::path, PathHash> file_map =
      get_images(folder);
  for (const auto &file : file_map) {
    std::string f = file.first.string();
    apply_glcm_0(&f, true);
  }

  return 0;
}
