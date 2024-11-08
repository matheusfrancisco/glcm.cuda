#include "file.h"
#include "image.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <unordered_map>

using namespace std;
namespace fs = std::filesystem;

__global__ void glcm_cuda_0(int *matrix, int *glcm, int n_col, int n_row,
                            int glcm_max) {
  // Calculate the total number of possible pairs: (n_row) * (n_col -1)
  unsigned int total_pairs = n_row * (n_col - 1);
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total_pairs) {
    // Determine the row and column from idx
    int row = idx / (n_col - 1);
    int col = idx % (n_col - 1);

    // Calculate the linear indices for the pair
    int current_idx = row * n_col + col;
    int next_idx = current_idx + 1;

    // Retrieve gray levels
    int current = matrix[current_idx];
    int next = matrix[next_idx];

    // Validate gray levels
    if (current >= 0 && current < glcm_max && next >= 0 && next < glcm_max) {
      // Compute the GLCM index
      int k = current * glcm_max + next;

      // Atomically increment the GLCM count
      atomicAdd(&glcm[k], 1);
    }
  }
}

void checkCudaError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error after %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void apply_glcm_1(std::string *file, bool write_output = false) {

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
  std::cout << "max: " << max << std::endl;

  int n_row = image_png.height;
  int n_col = image_png.width;

  int glcm_size = (max * max) * sizeof(int);
  int *h_glcm_cuda = (int *)malloc(glcm_size);

  if (matrix == NULL || h_glcm_cuda == NULL) {
    std::cerr << "Error allocating memory" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Compare CPU and CUDA GLCMs

  memset(h_glcm_cuda, 0, sizeof(glcm_size));

  // Allocate device memory
  int *d_matrix, *d_glcm;
  cudaMalloc((void **)&d_matrix, sizeof(int) * n_row * n_col);
  checkCudaError("cudaMalloc d_matrix");
  cudaMalloc((void **)&d_glcm, sizeof(int) * max * max);
  checkCudaError("cudaMalloc d_glcm");

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix, sizeof(int) * n_row * n_col,
             cudaMemcpyHostToDevice);
  checkCudaError("cudaMemcpy to d_matrix");

  // Initialize GLCM on device to zero
  cudaMemset(d_glcm, 0, sizeof(int) * max * max);
  checkCudaError("cudaMemset d_glcm");

  // Define CUDA kernel launch parameters
  int threads_per_block = 256;
  int total_pairs = n_row * (n_col - 1);
  int number_of_blocks =
      (total_pairs + threads_per_block - 1) / threads_per_block;

  //// Launch the CUDA kernel
  glcm_cuda_0<<<number_of_blocks, threads_per_block>>>(d_matrix, d_glcm, n_col,
                                                       n_row, max);
  checkCudaError("glcm_cuda_optimized kernel launch");

  // Synchronize to ensure kernel completion
  cudaDeviceSynchronize();
  checkCudaError("cudaDeviceSynchronize");

  // Copy GLCM back to host
  cudaMemcpy(h_glcm_cuda, d_glcm, sizeof(int) * max * max,
             cudaMemcpyDeviceToHost);
  checkCudaError("cudaMemcpy to h_glcm_cuda");

  if (write_output) {
    std::string r;
    {
      fs::path file_path(file->c_str());
      fs::path new_file_name =
          "result/" + file_path.stem().string() + "_gpu_result.txt";
      fs::path new_file_path = file_path.parent_path() / new_file_name;
      r = new_file_path.string();
    }
    write_image_matrix(r, h_glcm_cuda, max, max);
  }

  // Cleanup
  cudaFree(d_matrix);
  cudaFree(d_glcm);
}

int main() {
  std::string folder = "data";

  std::unordered_map<fs::path, fs::path, PathHash> file_map =
      get_images(folder);

  for (const auto &file : file_map) {
    std::string f = file.first.string();
    apply_glcm_1(&f, true);
  }
  cudaDeviceSynchronize();

  return 0;
}
