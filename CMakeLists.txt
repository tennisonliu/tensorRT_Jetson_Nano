cmake_minimum_required(VERSION 3.5)
project(trt CXX)

add_definitions(-std=c++11)
include_directories(/usr/local/include/opencv4/opencv2)
include_directories(/usr/local/cuda/include)
include_directories(/home/wijayw/Desktop/tliu_trt/src)

link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/lib)

find_package(OpenCV 4.1.0 REQUIRED)

add_executable(mobilenet_trt mobilenet_trt.cpp src/tensorNet.h)

target_link_libraries(mobilenet_trt ${OpenCV_LIBS})
target_link_libraries(mobilenet_trt cudart)
target_link_libraries(mobilenet_trt nvinfer)
target_link_libraries(mobilenet_trt nvparsers)
