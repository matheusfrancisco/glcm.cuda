#include "DICOMReader.h"
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
                  std::string filename = "default", bool write_output = false) {

  std::cout << "max: " << max << std::endl;

  int dx_array[] = {1, 1, 0, -1, -1, -1, 0, -1};
  int dy_array[] = {0, -1, -1, -1, 0, 1, 1, 1};
  int num_directions = 8;

  int *d_matrix, *d_glcm[num_directions];

  int glcm_size = (max * max) * sizeof(int);

  if (matrix == NULL) {
    std::cerr << "Error allocating memory" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Compare CPU and CUDA GLCMs
  for (int i = 0; i < num_directions; i++) {
    std::cout << "CudaMalloc: " << i << std::endl;
    cudaMalloc(&d_glcm[i], glcm_size);
    checkCudaError("cudaMalloc d_glcm");
    cudaMemset(d_glcm[i], 0, glcm_size);
    checkCudaError("cudaMemset d_glcm");
  }

  // memset(h_glcm_cuda, 0, sizeof(glcm_size));

  // Allocate device memory
  cudaMalloc((void **)&d_matrix, sizeof(int) * n_row * n_col);
  checkCudaError("cudaMalloc d_matrix");

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix, sizeof(int) * n_row * n_col,
             cudaMemcpyHostToDevice);
  checkCudaError("cudaMemcpy to d_matrix");

  // Define CUDA kernel launch parameters
  int threads_per_block = 256;
  int total_pairs = n_row * (n_col - 1);
  int number_of_blocks =
      (total_pairs + threads_per_block - 1) / threads_per_block;

  for (int dir = 0; dir < num_directions; dir++) {
    int dx = dx_array[dir];
    int dy = dy_array[dir];

    glcm_cuda_direction<<<number_of_blocks, threads_per_block>>>(
        d_matrix, d_glcm[dir], n_col, n_row, max, dx, dy);
    checkCudaError("glcm_cuda_optimized kernel launch");
  }

  // Synchronize to ensure kernel completion
  cudaDeviceSynchronize();
  checkCudaError("cudaDeviceSynchronize");

  // Copy GLCM back to host
  for (int i = 0; i < num_directions; i++) {
    //  int *h_glcm_cuda = (int *)malloc(glcm_size);
    int *h_glcm_cuda = (int *)malloc((max * max) * sizeof(int));

    cudaMemcpy(h_glcm_cuda, d_glcm[i], sizeof(int) * (max * max),
               cudaMemcpyDeviceToHost);

    checkCudaError("cudaMemcpy to h_glcm_cuda");

    if (write_output) {
      std::string r;
      {
        std::cout << "Writing output: " << filename.c_str() << std::endl;
        std::string path = filename;
        std::size_t last_slash = path.find_last_of("/\\");
        std::string file_path = path.substr(last_slash + 1);

        std::string new_file_name =
            "/home/chico/m/chico/glcm.cuda/data/result/" + file_path + "_" +
            std::to_string(i) + "_gpu_result.txt";
        std::cout << "Writing output: " << new_file_name << std::endl;
        r = new_file_name.c_str();
      }
      write_image_matrix(r, h_glcm_cuda, max, max);
    }

    // Cleanup
    cudaFree(d_glcm[i]);
    free(h_glcm_cuda);
  }
  cudaFree(d_matrix);
}

int main() {
  std::string folder = "/home/chico/m/chico/glcm.cuda/data";

  std::unordered_map<fs::path, fs::path, PathHash> file_map =
      get_images(folder);

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
    apply_glcm_1(matrix, max, image_png.height, image_png.width, f, true);
    std::cout << "done" << std::endl;
  }

  DICOMImage image;

  std::string folder_dcm = "/home/chico/m/chico/glcm.cuda/dataset";

  std::unordered_map<fs::path, fs::path, PathHash> file_map2 =
      get_images(folder_dcm);

  for (const auto &file : file_map2) {

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
      apply_glcm_1(matrix, max, image.rows, image.cols, file.first.string(),
                   true);

    } else {
      std::cerr << "Failed to read DICOM file." << std::endl;
      continue;
    }
  }
  cudaDeviceSynchronize();

  return 0;
}
