cmake_minimum_required(VERSION 3.23)
project(minitensor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # required to link a static lib into a Python module
set(CMAKE_CUDA_ARCHITECTURES 89)      # compile for the GPU in this machine

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit REQUIRED)
# Detect the installed nanobind package and import it into CMake
execute_process(
COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)





add_library(tensor_core STATIC tensor.cu)
target_link_libraries(tensor_core PUBLIC CUDA::cudart)
target_include_directories(tensor_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

nanobind_add_module(minitensor bindings.cpp)
target_link_libraries(minitensor PRIVATE tensor_core)
# Put the .pyd next to the Python scripts so `import minitensor` works from the project root
set_target_properties(minitensor PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})


add_executable(test_main tensor_test_main.cpp)
target_link_libraries(test_main PRIVATE tensor_core)
