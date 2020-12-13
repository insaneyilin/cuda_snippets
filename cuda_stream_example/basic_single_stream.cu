/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include "../common/common.h"

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

__global__ void kernel(int *a, int *b, int *c) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    // compute an average of three values in a and
    // three values in b
    int idx1 = (idx + 1) % 256;
    int idx2 = (idx + 2) % 256;
    float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
    float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
    // then compute the average of the two averages
    c[idx] = (as + bs) / 2;
  }
}

int main(int argc, char **argv) {
  cudaDeviceProp prop;
  int which_device;
  CHECK_CUDA_ERROR(cudaGetDevice(&which_device));
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, which_device));
  // device overlap is a feature that supports to simultaneously execute a
  // CUDA C kernel while performing a copy between device and host memory.
  if (!prop.deviceOverlap) {
    printf("Device will not handle overlaps, so no speed up from streams\n");
    return 0;
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsed_time;

  cudaStream_t stream;
  int *host_a;
  int *host_b;
  int *host_c;
  int *dev_a;
  int *dev_b;
  int *dev_c;

  // start the timers
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  // initialize the stream
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  // allocate the memory on the GPU
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

  // allocate host locked memory, used to stream
  CHECK_CUDA_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int),
                                 cudaHostAllocDefault));
  CHECK_CUDA_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int),
                                 cudaHostAllocDefault));
  CHECK_CUDA_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int),
                                 cudaHostAllocDefault));

  for (int i = 0; i < FULL_DATA_SIZE; ++i) {
    host_a[i] = rand();
    host_b[i] = rand();
  }
  CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
  // now loop over full data, in bite-sized chunks
  for (int i = 0; i < FULL_DATA_SIZE; i += N) {
    // copy the locked memory to the device, async
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dev_a, host_a + i,
                                     N * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dev_b, host_b + i,
                                     N * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream));
    // <<<grid dims, block dims, dynamic shared memory size, stream ID>>>
    kernel<<<N / 256, 256, 0, stream>>>(dev_a, dev_b, dev_c);

    // copy the data from device to locked memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_c + i, dev_c,
                                     N * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream));
  }
  // copy result chunk from locked to full buffer
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));

  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time,
                                        start, stop));
  printf("Time taken:  %3.1f ms\n", elapsed_time);

  // cleanup the streams and memory
  CHECK_CUDA_ERROR(cudaFreeHost(host_a));
  CHECK_CUDA_ERROR(cudaFreeHost(host_b));
  CHECK_CUDA_ERROR(cudaFreeHost(host_c));
  CHECK_CUDA_ERROR(cudaFree(dev_a));
  CHECK_CUDA_ERROR(cudaFree(dev_b));
  CHECK_CUDA_ERROR(cudaFree(dev_c));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

  return 0;
}
