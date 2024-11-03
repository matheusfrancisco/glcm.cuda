#include "image.h"
#include "lodepng/lodepng.h"

// void saveimagegray(const char *filename, rgb_image *img);
// bool isPngFormat(const std::string &filename);
//__global__ void setPixelToGrayscale(unsigned char *image, unsigned width,
//  unsigned height);
//

// check if the image is png format
bool isPngFormat(const std::string &filename) {
  return filename.find(".png") != std::string::npos;
}

void rgb_to_gray(png_image *img) {
  int number_of_elements = img->width * img->height;
  size_t size = number_of_elements * 4 * sizeof(unsigned char);
  for (int i = 0; i < img->width * img->height; i++) {
    unsigned char r = img->image[i * 4 + 0];
    unsigned char g = img->image[i * 4 + 1];
    unsigned char b = img->image[i * 4 + 2];
    unsigned char a = img->image[i * 4 + 3];
    unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    img->image[i * 4 + 0] = gray;
    img->image[i * 4 + 1] = gray;
    img->image[i * 4 + 2] = gray;
    img->image[i * 4 + 3] = a;
  }
}

void open_image_value_32b_array(const char *filename, png_image *img) {
  unsigned error;
  unsigned char *png;
  size_t pngsize;

  lodepng_load_file(&png, &pngsize, filename);

  error =
      lodepng_decode32(&img->image, &img->width, &img->height, png, pngsize);

  if (error) {
    std::cerr << "decoder error " << error << ": " << lodepng_error_text(error)
              << std::endl;
  }
}

// int main() {

// error = lodepng_decode32(&image_png.image, &image_png.width,
//                          &image_png.height, png, pngsize);

// int *glcm = (int *)malloc(sizeof(int) * (image_png.width *
// image_png.height));

// for (int i = 0; i < (image_png.width * image_png.height); i++) {
//   glcm[i] = image_png.image[i];
// }

//  std::vector<std::vector<unsigned char>> gray_matrix(
//      image_png.height, std::vector<unsigned char>(image_png.width));
//
// write_image_matrix(std::string("data/sample512.txt"), glcm,
// image_png.width,
//                    image_png.height);
//  lodepng_encode32_file("data/sample512_gray.png", image_png.image,
//                        image_png.width, image_png.height);
//}
