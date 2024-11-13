#include "file.h"
#include <fstream>

/**
 * @brief Get all the images in a folder
 *
 * @param folder
 * @return std::unordered_map<fs::path, fs::path, PathHash>
 */
std::unordered_map<fs::path, fs::path, PathHash>
get_images(const std::string folder) {

  std::unordered_map<fs::path, fs::path, PathHash> file_map;
  try {
    for (const auto &entry : fs::recursive_directory_iterator(folder)) {
      if (fs::is_regular_file(entry.path())) {
        fs::path filepath = entry.path();
        std::string extension = filepath.extension().string();
        if (extension == ".png" || extension == ".dcm") {
          std::cout << filepath << std::endl;
          file_map[filepath] = filepath;
        }
      }
    }

  } catch (const fs::filesystem_error &e) {
    std::cerr << "Filesystem error: " << e.what() << std::endl;
  }
  return file_map;
}

/**
 * @brief Write the image matrix to a file
 *
 * @param path
 * @param *matrix
 * @param number_of_rows
 * @param number_of_columns
 */
void write_image_matrix(std::string path, int *matrix, const int number_of_rows,
                        const int number_of_columns) {
  FILE *file = NULL;
  int *ic = matrix;
  file = fopen(path.c_str(), "w");
  if (file == NULL) {
    std::cout << "Error opening file" << std::endl;
    exit(1);
  }
  for (int i = 0; i < number_of_rows; i++) {
    for (int j = 0; j < number_of_columns; j++) {
      fprintf(file, "%d  ", ic[j]);
    }
    fprintf(file, "\n\n");
    ic += number_of_columns;
  }
  fclose(file);
  return;
}

void write_map_to_csv(const std::unordered_map<std::string, double> &time_map,
                      const std::string &filename) {
  std::ofstream csv_file(filename);
  if (!csv_file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  // Write the header
  csv_file << "string,time\n";

  // Write the map contents
  for (const auto &pair : time_map) {
    csv_file << pair.first << "," << std::fixed << std::setprecision(6)
             << pair.second << "\n";
  }

  csv_file.close();
}

void write_map_to_csv_cpu(
    const std::unordered_map<std::string, double> &time_map,
    const std::string &filename) {

  for (const auto &pair : time_map) {
    std::ofstream csv_file(filename + pair.first + ".csv");

    if (!csv_file.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
    }

    // Write the header
    csv_file << "string,time\n";
    // Write the map contents
    csv_file << pair.first << "," << std::fixed << std::setprecision(6)
             << pair.second << "\n";

    csv_file.close();
  }
}
