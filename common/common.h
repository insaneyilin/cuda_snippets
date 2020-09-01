/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#ifndef COMMON_COMMON_H_
#define COMMON_COMMON_H_

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

#endif  // COMMON_COMMON_H_

