project( cuda_stream_example )
cmake_minimum_required( VERSION 3.0 )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

include(../cmake/modules/Cuda10.cmake)

# we can use CUDA_ADD_EXECUTABLE() if there is a single main.cu file
CUDA_ADD_EXECUTABLE(basic_single_stream basic_single_stream.cu)

CUDA_ADD_EXECUTABLE(basic_two_streams_slow basic_two_streams_slow.cu)

CUDA_ADD_EXECUTABLE(basic_two_streams_correct basic_two_streams_correct.cu)
