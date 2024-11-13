// main.cpp

#include "../DICOMReader.h"
#include "../file.h"
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {

  std::string filename =
      "/home/chico/m/chico/glcm.cuda/dataset/ST000001/SE000001/IM000002.dcm";

  std::unordered_map<fs::path, fs::path, PathHash> file_map =
      get_images("/home/chico/m/chico/glcm.cuda/dataset");

  DICOMImage image;
  for (const auto &file : file_map) {
    std::string f = file.first.string();
    std::cout << f << std::endl;
  }

  if (readDICOMImage(filename, image)) {
    std::cout << "Patient's Name: " << image.patientName << std::endl;
    std::cout << "Image Dimensions: " << image.rows << " x " << image.cols
              << std::endl;
    std::cout << "Bits Allocated: " << image.bitsAllocated << std::endl;
    std::cout << "Samples Per Pixel: " << image.samplesPerPixel << std::endl;
    std::cout << "Pixel Data Size: " << image.pixelData.size() << std::endl;

    // Example: Accessing pixel data
    if (!image.pixelData.empty()) {
      std::cout << "First pixel intensity: " << image.pixelData[0] << std::endl;
    }

    // iterate in the image
    // for (int i = 0; i < image.rows * image.cols; i++) {
    //  printf("%d", image.pixelData[i]);
    //}

    // Optionally, process the image data as needed
    // For example, you can iterate over the pixelData vector
  } else {
    std::cerr << "Failed to read DICOM file." << std::endl;
    return 1;
  }

  return 0;
}
