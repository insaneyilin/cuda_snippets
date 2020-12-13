project( dot_product_with_atomic_lock )
cmake_minimum_required( VERSION 3.0 )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

include(../cmake/modules/Cuda10.cmake)

find_package(OpenCV REQUIRED)

CUDA_ADD_EXECUTABLE(dot_product_with_atomic_lock dot_product_with_atomic_lock.cu)
