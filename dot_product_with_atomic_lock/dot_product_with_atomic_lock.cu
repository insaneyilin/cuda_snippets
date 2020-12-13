/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include "../common/common.h"
#include "atomic_lock.h"

#define imin(a, b) (a < b ? a : b)

// need 5 blocks if we do not limit `blocks_per_grid`
const int N = (4 + 1) * 16;

// for dot_product reduction, `threads_per_block` must be a power of 2
const int threads_per_block = 16;

// Note the trick: `(N + (threads_per_block - 1)) / threads_per_block` is the
// the smallest multiple of `threads_per_block` that is greater than or equal
// to `N`.
// Here we 4 blocks at most.
const int blocks_per_grid =
    imin(4, (N + threads_per_block - 1) / threads_per_block);

__global__ void DotProduct(AtomicLock lock, float *a, float *b,
                           float *c) {
  // cache products of the corresponding entries of the two input arrays
  // cache is shared by all threads inside a block
  __shared__ float cache[threads_per_block];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cache_idx = threadIdx.x;

  float temp = 0;
  // we launch 32 blocks at most, so we use `while (tid < N)` loop here
  while (tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[cache_idx] = temp;

  // synchronize threads in this block
  __syncthreads();

  // for reductions, threads_per_block must be a power of 2
  // because of the following code
  int i = blockDim.x / 2;
  while (i != 0) {
      if (cache_idx < i) {
          cache[cache_idx] += cache[cache_idx + i];
      }
      // NOTE: We CANNOT put the following `__syncthreads();` to the `if` block
      // above, otherwise there would be "branch divergence".
      __syncthreads();
      i /= 2;
  }

  // Any `cache_idx` can (atomically) add `cache[0]` to global `c`
  // We choose `cache_idx == 0` for convenience
  if (cache_idx == 0) {
    lock.lock();
    *c += cache[0];
    lock.unlock();
  }
}

int main(int argc, char **argv) {
  float *a;
  float *b;
  float c = 0.f;
  float *dev_a;
  float *dev_b;
  float *dev_c;

  // allocate memory on the CPU side
  a = (float*)malloc(N * sizeof(float));
  b = (float*)malloc(N * sizeof(float));

  // allocate memory on the GPU side
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_c, sizeof(float)));

  // fill in the host memory with data
  float cpu_dot_prod = 0.f;
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i + 1;
    cpu_dot_prod += a[i] * b[i];
  }

  // copy input arrays value to GPU
  CHECK_CUDA_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float),
      cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float),
      cudaMemcpyHostToDevice));

  AtomicLock lock;
  DotProduct<<<blocks_per_grid, threads_per_block>>>(lock, dev_a, dev_b,
      dev_c);

  // copy array of partial sums back to CPU
  CHECK_CUDA_ERROR(cudaMemcpy(&c, dev_c,
      sizeof(float), cudaMemcpyDeviceToHost));

  float &dot_prod = c;

  // check results
  printf("blocks_per_grid: %d\n", blocks_per_grid);
  printf("threads_per_block: %d\n", threads_per_block);
  printf("Does GPU value %.10g = %.10g?\n", dot_prod, cpu_dot_prod);

  // free memory on the gpu side
  CHECK_CUDA_ERROR(cudaFree(dev_a));
  CHECK_CUDA_ERROR(cudaFree(dev_b));
  CHECK_CUDA_ERROR(cudaFree(dev_c));

  // free memory on the cpu side
  free(a);
  free(b);

  return 0;
}
