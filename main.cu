#include "file.h"
#include "image.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <unordered_map>

using namespace std;
namespace fs = std::filesystem;

__global__ void glcm_kernel(const int *matrix, int *glcm, int n_col, int n_row,
                            int glcm_max);

void check_cuda_error(const char *message) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error after %s: %s\n", message,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__global__ void glcm_nop(int *matrix, int *glcm, int n, int c, int max_v) {
  unsigned int idx = blockIdx.x * n + threadIdx.x;
  int i;
  int k = 0;
  for (i = 0; i < n; i++) {
    if (idx >= i * n && idx < ((i + 1) * n) - 1) {
      k = max_v * matrix[idx] + matrix[idx + 1];
      atomicAdd(&glcm[k], 1);
    }
  }
}

__global__ void glcm_optimized(int *matrix, int *glcm, int n, int max_v) {
  // Calculate the total number of possible pairs: (n rows) * (n-1 columns per
  // row)
  unsigned int total_pairs = n * (n - 1);
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total_pairs) {
    // Determine the row and column from idx
    int row = idx / (n - 1);
    int col = idx % (n - 1);

    // Calculate the linear indices for the pair
    int current = row * n + col;
    int next = current + 1;

    // Compute the GLCM index
    int k = max_v * matrix[current] + matrix[next];

    // Atomically increment the GLCM count
    atomicAdd(&glcm[k], 1);
  }
}

// CUDA Kernel for GLCM computation
__global__ void glcm_kernel(const int *matrix, int *glcm, int n_col, int n_row,
                            int glcm_max) {
  // Calculate the global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = n_row * (n_col - 1);

  if (idx < total_pairs) {
    // Determine the row and column from the thread index
    int row = idx / (n_col - 1);
    int col = idx % (n_col - 1);

    // Calculate the linear indices for the current and next pixels
    int current_idx = row * n_col + col;
    int next_idx = current_idx + 1;

    // Read the current and next pixel values using __ldg for read-only caching
    int current = __ldg(&matrix[current_idx]);
    int next = __ldg(&matrix[next_idx]);

    // Validate gray levels to prevent out-of-bounds access
    if (current >= 0 && current < glcm_max && next >= 0 && next < glcm_max) {
      // Compute the GLCM index
      int glcm_idx = current * glcm_max + next;

      // Atomically increment the GLCM entry
      atomicAdd(&glcm[glcm_idx], 1);
    }
  }
}

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

  max += 1;
  std::cout << "max: " << max << std::endl;

  int glcm_size = (max * max) * sizeof(int);
  int *glcm = (int *)malloc(glcm_size);

  if (matrix == NULL || glcm == NULL) {
    std::cerr << "Error allocating memory" << std::endl;
    exit(EXIT_FAILURE);
  }

  cudaMallocManaged(&matrix,
                    (image_png.width * image_png.height) * sizeof(int));

  // allocating device memory
  int *d_matrix, *d_glcm;
  cudaMalloc((void **)&d_matrix, m_size);
  check_cuda_error("allocating device memory: d_matrix");
  cudaMalloc((void **)&d_glcm, glcm_size);
  check_cuda_error("allocating device memory: d_glcm");

  // initializing device glcm to zero
  cudaMemset(d_glcm, 0, glcm_size);
  check_cuda_error("cudaMemset d_glcm");
  // copy the image matrix to the device
  cudaMemcpy(d_matrix, matrix, m_size, cudaMemcpyHostToDevice);
  check_cuda_error("cudaMemcpy matrix to d_matrix");

  dim3 blocks(image_png.width);
  dim3 grids((max + blocks.x - 1) / blocks.x, (max + blocks.y - 1) / blocks.y);

  // printf("blocks.x: %d\n", blocks.x);

  // printf("grids.x: %d\n", grids.x);
  // printf("grids.y: %d\n", grids.y);
  // printf("grids.z: %d\n", grids.z);

  // glcm_nop<<<grids, blocks>>>(matrix, glcm, image_png.height,
  // image_png.width,
  //                            max);

  // glcm_optimized<<<grids, blocks>>>(matrix, glcm, image_png.width, max);

  glcm_kernel<<<blocks, grids>>>(d_matrix, d_glcm, image_png.width,
                                 image_png.height, max);
  check_cuda_error("Kernel launch");

  // Wait for the kernel to finish
  cudaDeviceSynchronize();
  check_cuda_error("cudaDeviceSynchronize");
  cudaMemcpy(glcm, d_glcm, glcm_size, cudaMemcpyDeviceToHost);
  check_cuda_error("cudaMemcpy d_glcm to glcm");

  if (write_output) {
    std::string r;
    {
      fs::path file_path(file->c_str());
      fs::path new_file_name =
          "result/" + file_path.stem().string() + "gpu_result.txt";
      fs::path new_file_path = file_path.parent_path() / new_file_name;
      r = new_file_path.string();
    }
    write_image_matrix(r, glcm, max, max);
  }
}

int main() {
  std::string folder = "data";

  std::unordered_map<fs::path, fs::path, PathHash> file_map =
      get_images(folder);

  for (const auto &file : file_map) {
    std::string f = file.first.string();
    apply_glcm_0(&f, false);
  }
  cudaDeviceSynchronize();

  return 0;
}
