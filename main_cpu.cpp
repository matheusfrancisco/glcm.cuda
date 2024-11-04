#include "file.h"
#include "glcm.h"
#include "image.h"
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
  std::vector<int> matrix(image_png.height * image_png.width, 0);

  // get the maximum valur of the image
  int max = 0;
  for (int i = 0; i < (image_png.height * image_png.width); ++i) {
    matrix[i] = image_png.image[i];
    if (matrix[i] > max) {
      max = matrix[i];
    }
  }

  max += 1;
  std::vector<int> glcm(max * max, 0);
  // glcm_0_nop(matrix.data(), glcm.data(), image_png.height, image_png.width,
  //           max);

  glcm_optimized(matrix.data(), glcm.data(), image_png.height, image_png.width,
                 max);

  if (write_output) {
    std::string r;
    {
      fs::path file_path(file->c_str());
      fs::path new_file_name =
          "result/" + file_path.stem().string() + "_result.txt";
      fs::path new_file_path = file_path.parent_path() / new_file_name;
      r = new_file_path.string();
    }

    write_image_matrix(r, glcm.data(), max, max);
  }
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
