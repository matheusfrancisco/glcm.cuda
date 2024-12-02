
// DICOMReader.cpp

#include "DICOMReader.h"
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h> // For image access
#include <iostream>

// Function to read DICOM file and populate DICOMImage struct
bool readDICOMImage(const std::string &filename, DICOMImage &image) {

  // Load the DICOM file
  DcmFileFormat fileFormat;
  OFCondition status = fileFormat.loadFile(filename.c_str());

  if (!status.good()) {
    //   std::cerr << "Error: cannot read DICOM file (" << status.text() << ")"
    //   << std::endl;
    return false;
  }

  // Get the dataset
  DcmDataset *dataset = fileFormat.getDataset();

  // Extract Patient's Name
  OFString patientName;
  if (dataset->findAndGetOFString(DCM_PatientName, patientName).good()) {
    image.patientName = patientName.c_str();
  } else {
    image.patientName = "Patient's Name not found.";
  }

  // Extract Image Dimensions
  Uint16 rows, cols;
  if (dataset->findAndGetUint16(DCM_Rows, rows).good() &&
      dataset->findAndGetUint16(DCM_Columns, cols).good()) {
    image.rows = static_cast<int>(rows);
    image.cols = static_cast<int>(cols);
  } else {
    //  std::cerr << "Error: Image dimensions not found." << std::endl;
    return false;
  }

  // Extract Bits Allocated and Samples Per Pixel
  Uint16 bitsAllocated;
  Uint16 samplesPerPixel;
  if (dataset->findAndGetUint16(DCM_BitsAllocated, bitsAllocated).good() &&
      dataset->findAndGetUint16(DCM_SamplesPerPixel, samplesPerPixel).good()) {
    image.bitsAllocated = static_cast<int>(bitsAllocated);
    image.samplesPerPixel = static_cast<int>(samplesPerPixel);
  } else {
    // std::cerr << "Error: Bits Allocated or Samples Per Pixel not found." <<
    // std::endl;
    return false;
  }

  // Determine if pixel data is signed
  OFString pixelRepresentationStr;
  Sint16 pixelRepresentation = 0; // 0 = unsigned, 1 = signed
  if (dataset->findAndGetSint16(DCM_PixelRepresentation, pixelRepresentation)
          .good()) {
    image.isSigned = (pixelRepresentation == 1);
  } else {
    image.isSigned = false; // Default to unsigned
  }

  // Calculate number of pixels
  size_t numPixels =
      static_cast<size_t>(image.rows) * static_cast<size_t>(image.cols);

  // Handle pixel data based on Bits Allocated
  if (image.bitsAllocated == 16) {
    // 16-bit pixels
    const Uint16 *pixelData16 = nullptr;
    if (dataset->findAndGetUint16Array(DCM_PixelData, pixelData16).good()) {
      image.pixelData.reserve(numPixels);
      for (size_t i = 0; i < numPixels; ++i) {
        if (image.isSigned) {
          image.pixelData.push_back(static_cast<int16_t>(pixelData16[i]));
        } else {
          image.pixelData.push_back(static_cast<int16_t>(pixelData16[i]));
        }
      }
    } else {
      //    std::cerr << "Error: Unable to retrieve pixel data as Uint16 array."
      //    << std::endl;
      return false;
    }
  } else if (image.bitsAllocated == 8) {
    // 8-bit pixels
    const Uint8 *pixelData8 = nullptr;
    std::vector<int16_t> tempData;
    if (dataset->findAndGetUint8Array(DCM_PixelData, pixelData8).good()) {
      image.pixelData.reserve(numPixels);
      for (size_t i = 0; i < numPixels; ++i) {
        image.pixelData.push_back(static_cast<int16_t>(pixelData8[i]));
      }
    } else {
      //   std::cerr << "Error: Unable to retrieve pixel data as Uint8 array."
      //   << std::endl;
      return false;
    }
  } else {
    // std::cerr << "Error: Unsupported Bits Allocated: " << image.bitsAllocated
    // << std::endl;
    return false;
  }

  // Optionally, apply Rescale Slope and Intercept if present
  // This is common in modalities like CT
  double rescaleSlope = 1.0;
  double rescaleIntercept = 0.0;
  dataset->findAndGetFloat64(DCM_RescaleSlope, rescaleSlope);
  dataset->findAndGetFloat64(DCM_RescaleIntercept, rescaleIntercept);

  for (auto &pixel : image.pixelData) {
    pixel = static_cast<int16_t>(pixel * rescaleSlope + rescaleIntercept);
  }

  return true;
}
