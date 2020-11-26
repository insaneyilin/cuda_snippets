/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include "../common/common.h"

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

__global__ void DotProduct(float *a, float *b, float *partial_c) {
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

  // Any `cache_idx` can write `cache[0]` to `partial_c[blockIdx.x]`
  // We choose `cache_idx == 0` for convenience
  if (cache_idx == 0) {
    partial_c[blockIdx.x] = cache[0];
  }
}

int main(int argc, char **argv) {
  float *a;
  float *b;
  float *partial_c;
  float *dev_a;
  float *dev_b;
  float *dev_partial_c;
  float dot_prod = 0.f;

  // allocate memory on the CPU side
  a = (float*)malloc(N * sizeof(float));
  b = (float*)malloc(N * sizeof(float));
  partial_c = (float*)malloc(blocks_per_grid * sizeof(float));

  // allocate memory on the GPU side
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_partial_c,
      blocks_per_grid * sizeof(float)));

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

  DotProduct<<<blocks_per_grid, threads_per_block>>>(dev_a, dev_b,
      dev_partial_c);

  // copy array of partial sums back to CPU
  CHECK_CUDA_ERROR(cudaMemcpy(partial_c, dev_partial_c,
      blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost));

  // add up the partial sums to get the final dot product
  for (int i = 0; i < blocks_per_grid; ++i) {
    dot_prod += partial_c[i];
  }

  // check results
  printf("blocks_per_grid: %d\n", blocks_per_grid);
  printf("threads_per_block: %d\n", threads_per_block);
  printf("Does GPU value %.10g = %.10g?\n", dot_prod, cpu_dot_prod);

  // free memory on the gpu side
  CHECK_CUDA_ERROR(cudaFree(dev_a));
  CHECK_CUDA_ERROR(cudaFree(dev_b));
  CHECK_CUDA_ERROR(cudaFree(dev_partial_c));

  // free memory on the cpu side
  free(a);
  free(b);
  free(partial_c);

  return 0;
}
