#ifndef IMAGE_H
#define IMAGE_H
#include <iostream>
#include <vector>

typedef struct {
  unsigned width;
  unsigned height;
  unsigned char *image; // RGBA

} png_image;

std::vector<std::vector<unsigned char>>  to_matrix();
void rgb_to_gray(png_image *img, unsigned char *img_gray);
void open_image_value_32b_array(const char *filename, png_image *img);

#endif // IMAGE_H
