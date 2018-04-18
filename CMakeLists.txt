cmake_minimum_required(VERSION 3.5)
project(FFTW-Example CXX)

set(CMAKE_CXX_STANDARD 14) # C++14 beacuse it's 2017...
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_BUILD_TYPE DEBUG)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")

# Load FFTW3
set(FFTW_USE_STATIC_LIBS ON)
include(FindFFTW)

# Get file list
set(SOURCE_FILES fftw_wrappers.cpp convolution_example.cpp)
set(HEADER_FILES ffftw_wrappers.hpp)

# Create the executable
add_executable(FFTW-Example ${HEADER_FILES} ${SOURCE_FILES})

if(FFTW_FOUND)
    target_link_libraries(PixelProcessor PRIVATE ${FFTW_LIBRARIES})
    target_include_directories(PixelProcessor PRIVATE ${FFTW_INCLUDES})
else()
	message(FATAL_ERROR "Unable to find FFTW3. Please install it and add it to your path.")
endif()