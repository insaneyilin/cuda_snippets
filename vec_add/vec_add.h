/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#ifndef VEC_ADD_VEC_ADD_H_
#define VEC_ADD_VEC_ADD_H_

#include <stdio.h>
#include <cuda_runtime.h>

static void CheckCudaError(cudaError_t err,
                           const char *file,
                           int line ) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA_ERROR(err) (CheckCudaError(err, __FILE__, __LINE__))

void cudaVecAdd(int vec_len, int *in_v1, int *in_v2, int *out_v);

#endif  // VEC_ADD_VEC_ADD_H_

