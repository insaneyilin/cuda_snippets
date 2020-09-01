project( cuda_vec_add_demo )
cmake_minimum_required( VERSION 3.0 )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

include(../cmake/modules/Cuda10.cmake)

# we can use CUDA_ADD_EXECUTABLE() if there is a single main.cu file
#CUDA_ADD_EXECUTABLE(main main.cu)

# specify c++ src files
set( CPP_SRCS main.cc )

# specify cuda c src files
set( CUDA_C_SRCS vec_add.cu )

# compile cuda obj
cuda_compile( CUDA_OBJ ${CUDA_C_SRCS})

add_executable( ${PROJECT_NAME} ${CPP_SRCS} ${CUDA_OBJ} )

# CUDA_SHARED_LIBS is defined in ${repo}/cmake/modules/Cuda10.cmake
target_link_libraries( ${PROJECT_NAME} ${CUDA_SHARED_LIBS} )

