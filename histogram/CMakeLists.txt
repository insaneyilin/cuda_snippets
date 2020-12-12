project( histogram_demo )
cmake_minimum_required( VERSION 3.0 )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

include(../cmake/modules/Cuda10.cmake)

# we can use CUDA_ADD_EXECUTABLE() if there is a single main.cu file
CUDA_ADD_EXECUTABLE(histogram_cpu histogram_cpu.cu)

CUDA_ADD_EXECUTABLE(histogram_gpu_global_mem_atomics histogram_gpu_global_mem_atomics.cu)

CUDA_ADD_EXECUTABLE(histogram_gpu_shared_mem_atomics histogram_gpu_shared_mem_atomics.cu)
