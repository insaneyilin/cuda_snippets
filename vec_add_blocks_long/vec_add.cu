/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */
#include "vec_add.h"

__global__  void vec_add_kernel(int vec_len, int *in_v1, int *in_v2,
                                int *out_v) {
  int t_id = blockDim.x * blockIdx.x + threadIdx.x;
  while (t_id < vec_len) {
    out_v[t_id] = in_v1[t_id] + in_v2[t_id];
    // (number of threads per block) * (number of blocks in the grid)
    // e.g. vec_add_kernel<<<128, 64>>>(...)
    //     => gridDim.x == 128, blockDim.x == 64
    t_id += blockDim.x * gridDim.x;
  }
}

void cudaVecAdd(int num_blocks, int num_threads, int vec_len,
    int *in_v1, int *in_v2, int *out_v) {
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
  // 128 block, 128 threads
  // be careful that we pass "device" pointers here
  vec_add_kernel<<<num_blocks, num_threads>>>(vec_len, d_in_v1, d_in_v2,
      d_out_v);

  // copy results back to host
  err = cudaMemcpy(out_v, d_out_v, sizeof(int) * vec_len,
                   cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR(err);

  // free the memory allocated on GPU
  CHECK_CUDA_ERROR(cudaFree(d_in_v1));
  CHECK_CUDA_ERROR(cudaFree(d_in_v2));
  CHECK_CUDA_ERROR(cudaFree(d_out_v));
}

