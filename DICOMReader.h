
// DICOMReader.h

#ifndef DICOMREADER_H
#define DICOMREADER_H

#include <string>
#include <vector>

// Struct to hold DICOM image information
struct DICOMImage {
    std::string patientName;
    int rows;
    int cols;
    int bitsAllocated;
    int samplesPerPixel;
    bool isSigned;
    std::vector<int16_t> pixelData; // Adjust type based on Bits Allocated
};

// Function to read DICOM file and populate DICOMImage struct
bool readDICOMImage(const std::string& filename, DICOMImage& image);

#endif // DICOMREADER_H
