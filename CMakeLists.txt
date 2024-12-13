cmake_minimum_required(VERSION 3.10)
project(glcm_cuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 70)

# Find DCMTK package
find_package(DCMTK REQUIRED COMPONENTS dcmdata)

if(NOT DCMTK_FOUND)
    message(FATAL_ERROR "DCMTK not found. Please install DCMTK.")
endif()

# Find OpenMP package
find_package(OpenMP REQUIRED)

if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP not found. Please install OpenMP.")
endif()

# Suppress all warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Disable all warnings in GCC/Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fopenmp")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -fopenmp")
elseif(MSVC)
    # Disable all warnings in MSVC
    add_compile_options(/W0)
endif()

# Include directories
include_directories(${DCMTK_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/lodepng)

# Add executable with all source files
add_executable(glcm_cuda
    main.cu
    file.cpp
    image.cpp
    glcm.cpp
    glcm_gpu.cu
    features.cu
    DICOMReader.cpp
    lodepng/lodepng.cpp
)

# Link DCMTK libraries and OpenMP to the executable
target_link_libraries(glcm_cuda ${DCMTK_LIBRARIES} OpenMP::OpenMP_CXX)

# Set target properties
set_target_properties(glcm_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# Require C++17 features
target_compile_features(glcm_cuda PUBLIC cxx_std_17)

# Optionally, add a 'run' target to execute the program after building
add_custom_target(run
    COMMAND glcm_cuda
    DEPENDS glcm_cuda
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

# Target 2: CPU-only executable
add_executable(glcm_cpu
    main_cpu.cpp
    file.cpp
    image.cpp
    glcm.cpp
    DICOMReader.cpp
    lodepng/lodepng.cpp
)

# Link DCMTK libraries and OpenMP to the CPU executable
target_link_libraries(glcm_cpu ${DCMTK_LIBRARIES} OpenMP::OpenMP_CXX)

# Set target properties for CPU
set_target_properties(glcm_cpu PROPERTIES
    CXX_STANDARD 17
)

# Require C++17 features for both targets
target_compile_features(glcm_cuda PUBLIC cxx_std_17)
target_compile_features(glcm_cpu PUBLIC cxx_std_17)

# Custom run target for CPU
add_custom_target(run_cpu
    COMMAND glcm_cpu
    DEPENDS glcm_cpu
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
