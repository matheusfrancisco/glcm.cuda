#include "DICOMReader.h"
#include "features.h"
#include "file.h"
#include "glcm_gpu.h"
#include "image.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>

using namespace std;
namespace fs = std::filesystem;

void checkCudaError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error after %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void apply_glcm_1(int *matrix, int max, int n_row, int n_col,
                  std::string result_csv, std::string filename = "default",
                  bool write_output = false) {

  std::cout << filename << std::endl;
  int dx_array[] = {0, 1, 1, 1, 0, -1, -1, -1};
  int dy_array[] = {1, 1, 0, -1, -1, -1, 0, 1};
  int num_directions = 8;

  int *d_matrix;

  // Define CUDA kernel launch parameters

  cudaMalloc((void **)&d_matrix, sizeof(int) * n_row * n_col);
  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix, sizeof(int) * n_row * n_col,
             cudaMemcpyHostToDevice);

  std::unordered_map<std::string, double> time_map;
  std::unordered_map<std::string, double> total_gpu;
  auto start_time_global = std::chrono::high_resolution_clock::now();

  std::vector<int *> h_glcm_cuda_vec(num_directions, nullptr);
  std::vector<float *> h_glcm_cuda_vec2(num_directions, nullptr);

  std::vector<string> degree(8);
  int glcm_size = (max * max) * sizeof(int);

  for (int dir = 0; dir < num_directions; dir++) {

    int *d_glcm;
    int dx = dx_array[dir];
    int dy = dy_array[dir];
    std::cout << "Direction: " << dir << " dx: " << dx << " dy: " << dy
              << std::endl;

    // std::cout << "CudaMalloc: " << dir << std::endl;
    cudaMalloc(&d_glcm, glcm_size);
    checkCudaError("cudaMalloc d_glcm");
    cudaMemset(d_glcm, 0, glcm_size);

    auto start_time = std::chrono::high_resolution_clock::now();
    checkCudaError("cudaMemset d_glcm");

    int threads_per_block = 256;
    int total_pairs = n_row * (n_col - 1);
    int number_of_blocks =
        (total_pairs + threads_per_block - 1) / threads_per_block;
    if (dx == 0 and dy == 1) { // 0
      // std::cout << "glcm_calculation_nol" << std::endl;
      // std::cout << "n_row: " << n_row << " n_col: " << n_col << std::endl;

      glcm_cuda_direction<<<n_col, n_row>>>(d_matrix, d_glcm, n_col, n_row, max,
                                            dy, dx);

      degree[0] = "0";
    }

    else if (dx == 1 and dy == 1) { // 45 degree
      glcm_cuda_direction<<<number_of_blocks, threads_per_block>>>(
          d_matrix, d_glcm, n_col, n_row, max, dy, dx);

      degree[1] = "45";

    } else if (dx == 1 and dy == 0) // 90
    {
      glcm_cuda_direction<<<number_of_blocks, threads_per_block>>>(
          d_matrix, d_glcm, n_col, n_row, max, dy, dx);
      degree[2] = "90";
    }

    else if (dx == 1 and dy == -1) // 135 degree
    {
      glcm_cuda_direction<<<n_col, n_row>>>(d_matrix, d_glcm, n_col, n_row, max,
                                            dy, dx);

      degree[3] = "135";
    } else if (dx == 0 and dy == -1) { //  180

      glcm_cuda_direction<<<number_of_blocks, threads_per_block>>>(
          d_matrix, d_glcm, n_col, n_row, max, dy, dx);

      degree[4] = "180";
    }

    else if (dx == -1 and dy == -1) // 225
    {
      glcm_cuda_direction<<<n_col, n_row>>>(d_matrix, d_glcm, n_col, n_row, max,
                                            dy, dx);
      degree[5] = "225";
    }

    else if (dx == -1 and dy == 0) { // 270 degree

      glcm_cuda_direction<<<number_of_blocks, threads_per_block>>>(
          d_matrix, d_glcm, n_col, n_row, max, dy, dx);

      degree[6] = "270";
    } else if (dx == -1 and dy == 1) // 315
    {
      glcm_cuda_direction<<<n_col, n_row>>>(d_matrix, d_glcm, n_col, n_row, max,
                                            dy, dx);

      degree[7] = "315";
    }

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    checkCudaError("cudaDeviceSynchronize");
    // Copy GLCM back to host
    int *h_glcm_cuda = (int *)malloc((max * max) * sizeof(int));
    cudaMemcpy(h_glcm_cuda, d_glcm, sizeof(int) * (max * max),
               cudaMemcpyDeviceToHost);

    checkCudaError("cudaMemcpy to h_glcm_cuda");
    int enabled_normalization = 0;

    if (enabled_normalization == 1) {
      // #TODO normaization the glcm for be more easy to calculate some features

      // needs to normalize the glcm matrix
      int sum = 0;
      for (int i = 0; i < max * max; i++) {
        sum += h_glcm_cuda[i];
      }
      // std::cout << "sum elements: " << sum << std::endl;

      int *d_g_glcm;
      cudaMalloc((void **)&d_g_glcm, sizeof(int) * max * max);
      //  Copy matrix to device
      cudaMemcpy(d_g_glcm, h_glcm_cuda, sizeof(int) * max * max,
                 cudaMemcpyHostToDevice);

      checkCudaError("move glcm from thos to device");

      float *h_glcm_cuda_normalized;
      cudaMallocManaged(&h_glcm_cuda_normalized, max * max * sizeof(float));
      checkCudaError("malloc glcm for cuda normalized");

      cudaMemset(h_glcm_cuda_normalized, 0.f, max * max * sizeof(float));
      checkCudaError("set normalized");

      norm<<<256, 256>>>(d_g_glcm, h_glcm_cuda_normalized, max, sum);

      cudaDeviceSynchronize();
      float *normalized = (float *)malloc(max * max * sizeof(float));

      cudaMemcpy(normalized, h_glcm_cuda_normalized, sizeof(float) * max * max,
                 cudaMemcpyDeviceToHost);

      checkCudaError("Copy normalized glcm");

      // copy normalized matrix to device again

      float *normalized_glcm;
      cudaMallocManaged((void **)&normalized_glcm, (max * max) * sizeof(float));
      checkCudaError("malloc normalized");

      // Copy matrix to device
      cudaMemcpy(normalized_glcm, normalized, sizeof(float) * (max * max),
                 cudaMemcpyHostToDevice);

      // extracting features from normalized glcm matrix
      // extracting contrast
      // float *contrast_value;
      // cudaMallocManaged(&contrast_value, (max * max) * sizeof(float));
      // checkCudaError("Initialize contrast");

      //
      // dim3 blockSize(32, 32);
      // dim3 gridSize((max + blockSize.x - 1) / blockSize.x, (max +
      // blockSize.y
      // - 1) / blockSize.y);
      //// features
      // contrast<<<gridSize, blockSize>>>(normalized_glcm, contrast_value,
      // max); cudaDeviceSynchronize(); printf("Contrast: %.4f\n",
      // contrast_value[0]);
      h_glcm_cuda_vec2[dir] = normalized_glcm;
    } else {

      h_glcm_cuda_vec[dir] = h_glcm_cuda;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    time_map[filename + "_" + std::to_string(dir)] = elapsed.count();
    // std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    cudaFree(d_glcm);
  }

  auto end_time_global = std::chrono::high_resolution_clock::now();
  total_gpu["total_gpu_dcm"] =
      std::chrono::duration<double>(end_time_global - start_time_global)
          .count();

  if (write_output) {
    for (int dir = 0; dir < num_directions; dir++) {

      // std::cout << "dir: " << dir << std::endl;
      // std::cout << "Writing output dir: " << degree[dir] << std::endl;
      std::string r;
      {
        // std::cout << "Writing output: " << filename.c_str() << std::endl;
        std::string path = filename;
        std::size_t last_slash = path.find_last_of("/\\");
        std::size_t second_last_slash =
            path.find_last_of("/\\", last_slash - 1);
        std::string file_path = path.substr(last_slash + 1);

        std::string part1 = path.substr(second_last_slash + 1,
                                        last_slash - second_last_slash - 1);
        std::string part2 = path.substr(last_slash + 1, path.find_last_of('.') -
                                                            last_slash - 1);

        std::string new_file_name =
            "/home/chico/m/chico/glcm.cuda/data/result/" + part1 + "-" + part2 +
            "_" + std::to_string(dir) + "_" + degree[dir] + "_gpu_result.txt";
        std::cout << "Writing output: " << new_file_name << std::endl;
        r = new_file_name.c_str();
      }

      write_image_matrix_glcm(r, h_glcm_cuda_vec[dir], max, max);
      int enabled_normalization = 0;
      if (enabled_normalization == 1) {
        write_image_matrix(r, h_glcm_cuda_vec2[dir], max, max);
      }
      free(h_glcm_cuda_vec[dir]);
    }
  }

  // write_map_to_csv(time_map, result_csv);
  // write_map_to_csv(total_gpu, "../total_gpu.csv");

  cudaFree(d_matrix);
}

int main() {
  std::string folder = "/home/chico/m/chico/glcm.cuda/data";

  std::string folder_dcm = "/home/chico/m/chico/glcm.cuda/dataset";

  std::unordered_map<fs::path, fs::path, PathHash> file_map =
      get_images(folder);

  std::unordered_map<fs::path, fs::path, PathHash> file_map2 =
      get_images(folder_dcm);

  int test_flag = 0;

  if (test_flag == 1) {

    int test_flag_2 = 0;
    if (test_flag_2 == 1) {
      png_image image_png;
      string f = "/home/chico/m/chico/glcm.cuda/data/sample1024.png";
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
      // nx width
      // ny  height
      apply_glcm_1(matrix, max, image_png.width, image_png.height,
                   "../data/csv_result/png_result.csv", f, true);
    } else {
      // auto file = file_map2.begin();

      //    string file =
      //    "/home/chico/m/chico/glcm.cuda/dataset/ST000001/SE000007/IM0000033.dcm";
      string file = "/home/chico/m/chico/glcm.cuda/dataset/ST000001/SE000007/"
                    "IM000016s.dcm";
      DICOMImage image;

      // std::cout << "Reading DICOM file: " << file.first.string() <<
      // std::endl;
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

        max += 1;
        std::cout << "max: " << max << std::endl;
        if (max < 10000) {
          std::string r =
              "../data/csv_result/dcm_result" + std::to_string(0) + ".csv";
          apply_glcm_1(matrix, max, image.rows, image.cols, r, file, true);
        }

      } else {
        std::cerr << "Failed to read DICOM file." << std::endl;
      }
    }
  }

  else {
    for (const auto &file : file_map) {
      std::string f = file.first.string();
      std::cout << f.c_str() << std::endl;

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
      max += 2;
      apply_glcm_1(matrix, max, image_png.height, image_png.width,
                   "../data/csv_result/png_result.csv", f, true);
      std::cout << "done" << std::endl;
    }

    int count = 0;
    for (const auto &file : file_map2) {
      DICOMImage image;

      // std::cout << "Reading DICOM file: " << file.first.string() <<
      // std::endl;
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
          std::string r =
              "../data/csv_result/dcm_result" + std::to_string(count) + ".csv";
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
  cudaDeviceSynchronize();

  return 0;
}
