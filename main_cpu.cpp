#include "DICOMReader.h"
#include "file.h"
#include "glcm.h"
#include "image.h"
#include <cstring>
#include <filesystem>
#include <iostream>
#include <unordered_map>

using namespace std;
namespace fs = std::filesystem;

void apply_glcm_1(int *matrix, int max, int n_row, int n_col,
                  std::string result_csv, std::string filename = "default",
                  bool write_output = false) {

  auto directions = get_directions();
  std::unordered_map<std::string, double> time_map;
  std::unordered_map<std::string, double> total_cpu;

  auto start_time_global = std::chrono::high_resolution_clock::now();

  std::vector<string> degree(8);
  degree[0] = "0";
  degree[1] = "45";
  degree[2] = "90";
  degree[3] = "135";
  degree[4] = "180";
  degree[5] = "225";
  degree[6] = "270";
  degree[7] = "315";

  int count = 0;
  for (const auto &dir : directions) {
    std::cout << "Direction: " << dir.first << " " << dir.second << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    int glcm_size = (max * max) * sizeof(int);
    int *r_glcm = (int *)malloc(glcm_size);

    memset(r_glcm, 0, glcm_size);

    glcm_directions(matrix, r_glcm, n_col, n_row, max, dir.first, dir.second);
    int sum = 0;

    int enabled_normalization = 0;
    if (enabled_normalization == 1) {
      float *normalized = new float[max * max];
      memset(normalized, 0, (max * max) * sizeof(float));
      for (int i = 0; i < max * max; i++) {
        sum += r_glcm[i];
      }
      // std::cout << "my sum:" << sum << " ";
      norm_cpu(r_glcm, normalized, max, sum);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    // std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    if (write_output) {
      std::string r;
      {
        std::cout << "Writing output: " << filename.c_str() << std::endl;
        std::string path = filename;
        std::size_t last_slash = path.find_last_of("/\\");
        std::size_t second_last_slash =
            path.find_last_of("/\\", last_slash - 1);
        std::string file_path = path.substr(last_slash + 1);

        std::string part1 = path.substr(second_last_slash + 1,
                                        last_slash - second_last_slash - 1);
        std::string part2 = path.substr(last_slash + 1, path.find_last_of('.') -
                                                            last_slash - 1);

        time_map[part1 + "-" + part2 + "_" + std::to_string(dir.first)] =
            elapsed.count();
        std::string new_file_name =
            "/home/chico/m/chico/glcm.cuda/data/result_cpu/" + part1 + "-" +
            part2 + "_" + std::to_string(dir.first) + "_"+ degree[count]+"_result.txt";
        std::cout << "Writing output: " << new_file_name << std::endl;
        r = new_file_name.c_str();
      }
      // write_image_matrix(r, normalized, max , max );
      write_image_matrix_glcm(r, r_glcm, max, max);
    }
    free(r_glcm);
    //    free(normalized);
    count += 1;
  }
  free(matrix);
  auto end_time_global = std::chrono::high_resolution_clock::now();
  total_cpu["total_cpu"] =
      std::chrono::duration<double>(end_time_global - start_time_global)
          .count();

  std::cout << "Total elapsed time: "
            << std::chrono::duration<double>(end_time_global -
                                             start_time_global)
                   .count()
            << " seconds\n";
  write_map_to_csv_cpu(time_map, result_csv);
  write_map_to_csv(total_cpu, "../total_cpu.csv");
}

int main() {
  std::string folder = "data";

  std::unordered_map<fs::path, fs::path, PathHash> file_map =
      get_images(folder);

  std::string folder_dcm = "/home/chico/m/chico/glcm.cuda/dataset";

  std::unordered_map<fs::path, fs::path, PathHash> file_map2 =
      get_images(folder_dcm);

  int test_flag = 1;

  // this was only for run with one image and test it
  if (test_flag == 0) {

    string file =
        "/home/chico/m/chico/glcm.cuda/dataset/ST000001/SE000007/IM0000033.dcm";

    DICOMImage image;

    if (readDICOMImage(file, image)) {
      std::cout << "Image Dimensions: " << image.rows << " x " << image.cols
                << std::endl;

      // Example: Accessing pixel data
      if (!image.pixelData.empty()) {
        std::cout << "First pixel intensity: " << image.pixelData[0]
                  << std::endl;
      }

      int *matrix = (int *)malloc((image.rows * image.cols) * sizeof(int));
      int max = 0;
      for (int i = 0; i < image.rows * image.cols; i++) {
        matrix[i] = image.pixelData[i];
        if (image.pixelData[i] > max) {
          max = image.pixelData[i];
        }
      }

      if (max < 10000) {
        std::string r =
            "../data/csv_result_cpu/cpu_result" + std::to_string(0) + ".csv";
        apply_glcm_1(matrix, max, image.rows, image.cols, r, file, true);
      }

    } else {
      std::cerr << "Failed to read DICOM file." << std::endl;
    }
  } else {

    for (const auto &file : file_map) {
      std::string f = file.first.string();

      png_image image_png;
      std::cout << f.c_str() << std::endl;

      // open the image png and put it into an array
      open_image_value_32b_array(f.c_str(), &image_png);

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

      max += 1;

      int n_row = image_png.height;
      int n_col = image_png.width;

      std::string result = "data/csv_result_cpu/cpu_result";
      apply_glcm_1(matrix, max, n_row, n_col, result, f, true);
    }

    int count = 0;
    for (const auto &file : file_map2) {
      DICOMImage image;

      std::cout << "Reading DICOM file: " << file.first.string() << std::endl;
      if (readDICOMImage(file.first.string(), image)) {
        std::cout << "Image Dimensions: " << image.rows << " x " << image.cols
                  << std::endl;

        // Example: Accessing pixel data
        if (!image.pixelData.empty()) {
          std::cout << "First pixel intensity: " << image.pixelData[0]
                    << std::endl;
        }

        int *matrix = (int *)malloc((image.rows * image.cols) * sizeof(int));
        int max = 0;
        for (int i = 0; i < image.rows * image.cols; i++) {
          matrix[i] = image.pixelData[i];
          if (image.pixelData[i] > max) {
            max = image.pixelData[i];
          }
        }
        max += 1;
        if (max < 10000) {
          std::string r = "../data/csv_result_cpu/cpu_result" +
                          std::to_string(count) + ".csv";
          apply_glcm_1(matrix, max, image.rows, image.cols, r,
                       file.first.string(), true);
        }

      } else {
        std::cerr << "Failed to read DICOM file." << std::endl;
        continue;
      }
      count++;
    }
  }
  return 0;
}
