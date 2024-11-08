cmake_minimum_required(VERSION 3.10)
project(glcm_cuda)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find DCMTK package
find_package(DCMTK REQUIRED COMPONENTS dcmdata)

if(NOT DCMTK_FOUND)
    message(FATAL_ERROR "DCMTK not found. Please install DCMTK.")
endif()

# Include DCMTK headers
include_directories(${DCMTK_INCLUDE_DIRS})

# Add executable for main.cpp
add_executable(glcm_cuda main.cpp file.cpp DICOMReader.cpp glcm.cpp)

# Link DCMTK libraries to the executable
target_link_libraries(glcm_cuda ${DCMTK_LIBRARIES})
