cmake_minimum_required(VERSION 3.23)
project(tensor_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(${PROJECT_NAME}
    tensor.cu
    tensor.h
    tensor_test_main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 89)
