project( cuda_julia_set_demo )
cmake_minimum_required( VERSION 3.0 )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

include(../cmake/modules/Cuda10.cmake)

# we can use CUDA_ADD_EXECUTABLE() if there is a single main.cu file
CUDA_ADD_EXECUTABLE(julia_set_cpu julia_cpu.cu)

CUDA_ADD_EXECUTABLE(julia_set_gpu julia_gpu.cu)
