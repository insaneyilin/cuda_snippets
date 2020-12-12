/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include "../common/common.h"

#define SIZE (100*1024*1024)

__global__ void histo_kernel(unsigned char *buffer,
                             long size,
                             unsigned int *histo) {
  // clear out the accumulation buffer called temp
  // since we are launched with 256 threads, it is easy
  // to clear that memory with one write per thread
  __shared__  unsigned int temp[256];
  temp[threadIdx.x] = 0;
  __syncthreads();

  // calculate the starting index and the offset to the next
  // block that each thread will be processing
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while (i < size) {
    atomicAdd(&temp[buffer[i]], 1);
    i += stride;
  }
  // sync the data from the above writes to shared memory
  // then add the shared memory values to the values from
  // the other thread blocks using global memory
  // atomic adds
  // same as before, since we have 256 threads, updating the
  // global histogram is just one write per thread!
  __syncthreads();
  atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

int main(int argc, char **argv) {
  // allocate buffer with random values
  unsigned char *buffer =
      (unsigned char*)big_random_block(SIZE);

  for (int i = 0; i < SIZE; ++i) {
    buffer[i] = buffer[i] % 256;
  }

  unsigned int gpu_histo[256];
  for (int i = 0; i < 256; ++i) {
    gpu_histo[i] = 0;
  }

  // ------ GPU histogram ------
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

  unsigned char *dev_buffer;
  unsigned int *dev_histo;

  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_buffer, SIZE));
  CHECK_CUDA_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE,
                              cudaMemcpyHostToDevice));

  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_histo,
                              256 * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemset(dev_histo, 0,
                              256 * sizeof(int)));

  // kernel launch - 2x the number of mps gave best timing
  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount;
  histo_kernel<<<blocks * 2, 256>>>(dev_buffer, SIZE, dev_histo);

  CHECK_CUDA_ERROR(cudaMemcpy(gpu_histo, dev_histo,
                              256 * sizeof(int),
                              cudaMemcpyDeviceToHost));

  // get stop time, and display the timing results
  CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  float elapsed_time;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("GPU histogram time: %3.1f ms\n", elapsed_time);

  long histo_count = 0;
  for (int i = 0; i < 256; ++i) {
    histo_count += gpu_histo[i];
    // printf("%d ", histo[i]);
  }
  // printf("\n");
  printf("GPU Histogram Sum: %ld\n", histo_count);

  // ------ CPU histogram ------
  unsigned int cpu_histo[256];
  for (int i = 0; i < 256; ++i) {
    cpu_histo[i] = 0;
  }

  clock_t cpu_start;
  clock_t cpu_stop;
  cpu_start = clock();
  for (int i = 0; i < SIZE; ++i) {
    ++cpu_histo[buffer[i]];
  }
  cpu_stop = clock();
  float cpu_elapsed_time = (float)(cpu_stop - cpu_start) /
                      (float)CLOCKS_PER_SEC * 1000.0f;

  printf("CPU histogram time: %3.1f ms\n", cpu_elapsed_time);

  histo_count = 0;
  for (int i = 0; i < 256; ++i) {
    histo_count += cpu_histo[i];
    // printf("%d ", histo[i]);
  }
  // printf("\n");
  printf("CPU Histogram Sum: %ld\n", histo_count);

  for (int i = 0; i < 256; ++i) {
    if (gpu_histo[i] != cpu_histo[i]) {
      printf("ERROR! gpu histogram is different with cpu histogram\n");
    }
  }
  printf("gpu histogram is the same with cpu histogram\n");

  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  cudaFree(dev_histo);
  cudaFree(dev_buffer);
  free(buffer);

  return 0;
}

