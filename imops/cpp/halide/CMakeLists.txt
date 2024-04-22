cmake_minimum_required(VERSION 3.23)

# set(CMAKE_C_COMPILER /usr/local/gcc-13.2.0/bin/gcc)
# set(CMAKE_CXX_COMPILER /usr/local/gcc-13.2.0/bin/g++)

project(parinterp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_PREFIX_PATH "/homes/lbelkov/packages/Halide-17.0.1-x86-64-linux")
list(APPEND CMAKE_PREFIX_PATH "/homes/lbelkov/packages/pybind11/include/")
# list(APPEND CMAKE_PREFIX_PATH "~/miniconda3/envs/py311/lib/python3.11/site-packages/torch/include")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(/homes/lbelkov/packages/pybind11/include/)
# include_directories(~/miniconda3/envs/py311/lib/python3.11/site-packages/torch/include)
include_directories(/homes/lbelkov/packages/Halide-17.0.1-x86-64-linux/include/)

# find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Halide REQUIRED)

pybind11_add_module(contour_segments get_contour_segments_stacked_src.cpp)