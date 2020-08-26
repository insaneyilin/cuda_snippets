set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(CUDA 10.0 REQUIRED)
include_directories(SYSTEM ${CUDA_TOOLKIT_ROOT_DIR}/include/)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/)
list(APPEND CUDA_NVCC_FLAGS -std=c++11)
message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

set(CUDA_STATIC_LIBS
  libnppidei_static.a
  libnppicc_static.a
  libnppig_static.a
  libnppc_static.a
  libcudart_static.a libcurand_static.a
  libcudnn_static.a libculibos.a rt libcublas_static.a)
set(CUDA_SHARED_LIBS
  libnppidei.so
  libnppicc.so
  libnppig.so
  libnppc.so
  libcudart.so libcurand.so libculibos.so libcublas.so rt)
set(CUDA_LIBS ${CUDA_SHARED_LIBS})
