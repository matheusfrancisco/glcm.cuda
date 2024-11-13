// file.h
#ifndef FILE_H
#define FILE_H
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdio.h>
#include <unordered_map>

namespace fs = std::filesystem;

struct PathHash {
  std::size_t operator()(const fs::path &p) const {
    return std::hash<std::string>{}(p.string());
  }
};

/**
 * @brief Write the image matrix to a file
 *
 * @param path
 * @param *matrix
 * @param number_of_rows
 * @param number_of_columns
 */
void write_image_matrix(std::string path, int *matrix, const int number_of_rows,
                        const int number_of_columns);

/**
 * @brief Get all the images in a folder
 *
 * @param folder
 * @return std::unordered_map<fs::path, fs::path, PathHash>
 */
std::unordered_map<fs::path, fs::path, PathHash>
get_images(const std::string folder);

void write_map_to_csv(const std::unordered_map<std::string, double> &time_map,
                      const std::string &filename);
#endif // FILE_H
