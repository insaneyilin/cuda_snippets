project( ray_tracing )
cmake_minimum_required( VERSION 3.0 )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

include(../cmake/modules/Cuda10.cmake)

# we can use CUDA_ADD_EXECUTABLE() if there is a single main.cu file
CUDA_ADD_EXECUTABLE(ray_tracing_no_constant_mem ray_tracing_no_constant_mem.cu)

CUDA_ADD_EXECUTABLE(ray_tracing_constant_mem ray_tracing_constant_mem.cu)
