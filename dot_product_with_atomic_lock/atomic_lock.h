#ifndef _ATOMIC_LOCK_H_
#define _ATOMIC_LOCK_H_

#include "../common/common.h"

struct AtomicLock {
  int *mutex;
  AtomicLock() {
    CHECK_CUDA_ERROR(cudaMalloc((void**)&mutex, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(mutex, 0, sizeof(int)));
  }

  ~AtomicLock() {
    cudaFree(mutex);
  }

  __device__ void lock() {
    while (atomicCAS(mutex, 0, 1) != 0);
    __threadfence();
  }

  __device__ void unlock() {
    __threadfence();
    atomicExch(mutex, 0);
  }
};

#endif
