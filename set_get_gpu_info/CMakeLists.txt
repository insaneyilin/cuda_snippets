project( cuda_gpu_info_demo )
cmake_minimum_required( VERSION 3.0 )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

include(../cmake/modules/Cuda10.cmake)

# we can use CUDA_ADD_EXECUTABLE() if there is a single main.cu file
CUDA_ADD_EXECUTABLE(enum_gpu enum_gpu.cu)

CUDA_ADD_EXECUTABLE(set_gpu set_gpu.cu)

