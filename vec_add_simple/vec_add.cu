/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */
#include "vec_add.h"

__global__  void vec_add_kernel(int vec_len, int *in_v1, int *in_v2,
                                int *out_v) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < vec_len) {
    out_v[i] = in_v1[i] + in_v2[i];
  }
}

void cudaVecAdd(int vec_len, int *in_v1, int *in_v2, int *out_v) {
  cudaError_t err;

  // device memory pointers
  int *d_in_v1 = nullptr;
  int *d_in_v2 = nullptr;
  int *d_out_v = nullptr;

  // device memory allocation
  err = cudaMalloc((void **)&d_in_v1, sizeof(int) * vec_len);
  CHECK_CUDA_ERROR(err);

  err = cudaMalloc((void **)&d_in_v2, sizeof(int) * vec_len);
  CHECK_CUDA_ERROR(err);

  err = cudaMalloc((void **)&d_out_v, sizeof(int) * vec_len);
  CHECK_CUDA_ERROR(err);

  // copy memory to device
  err = cudaMemcpy(d_in_v1, in_v1, sizeof(int) * vec_len,
                   cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR(err);

  err = cudaMemcpy(d_in_v2, in_v2, sizeof(int) * vec_len,
                   cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR(err);

  // calling the kernel
  // 1 block, 32 threads
  // be careful that we pass "device" pointers here
  vec_add_kernel<<<1, 32>>>(vec_len, d_in_v1, d_in_v2, d_out_v);

  // copy results back to host
  err = cudaMemcpy(out_v, d_out_v, sizeof(int) * vec_len,
                   cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR(err);

  // free the memory allocated on GPU
  CHECK_CUDA_ERROR(cudaFree(d_in_v1));
  CHECK_CUDA_ERROR(cudaFree(d_in_v2));
  CHECK_CUDA_ERROR(cudaFree(d_out_v));
}

