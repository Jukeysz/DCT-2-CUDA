cmake_minimum_required(VERSION 3.10)
project(LightFieldProject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13")
set(CMAKE_CUDA_ARCHITECTURES 86)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV.")
endif()

# Find CUDA
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please install CUDA.")
endif()

# Add source files
set(SOURCE_FILES
    LightFieldLoader.cpp
    main.cpp
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add CUDA files
set(CUDA_SOURCE_FILES
    apply_dct1d_7x7x8x8.cu
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${CUDA_SOURCE_FILES})

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lnvToolsExt
        --fmad=false
    >
)

# CUDA settings
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES native)
