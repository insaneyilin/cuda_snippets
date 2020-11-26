project( ripple_anim )
cmake_minimum_required( VERSION 3.0 )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

include(../cmake/modules/Cuda10.cmake)

find_package(OpenCV REQUIRED)

CUDA_ADD_EXECUTABLE(ripple_anim ripple_anim.cu)
TARGET_LINK_LIBRARIES(ripple_anim ${OpenCV_LIBS})
