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

#define CHECK_NULL( a ) {if (a == NULL) { \
                        printf( "Host memory failed in %s at line %d\n", \
                        __FILE__, __LINE__ ); \
                        exit( EXIT_FAILURE );}}

void* big_random_block(int size) {
  unsigned char *data = (unsigned char*)malloc(size);
  CHECK_NULL(data);
  for (int i = 0; i < size; ++i) {
    data[i] = rand();
  }
  return data;
}

int* big_random_block_int( int size ) {
  int *data = (int*)malloc(size * sizeof(int));
  CHECK_NULL(data);
  for (int i = 0; i < size; ++i) {
    data[i] = rand();
  }
  return data;
}

#endif  // COMMON_COMMON_H_
