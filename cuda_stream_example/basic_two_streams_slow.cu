/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include "../common/common.h"

#define N (1024*1024)
#define FULL_DATA_SIZE (N * 100)

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

  cudaStream_t stream0;
  cudaStream_t stream1;
  int *host_a;
  int *host_b;
  int *host_c;
  // device memory for stream0
  int *dev_a0;
  int *dev_b0;
  int *dev_c0;
  // device memory for stream1
  int *dev_a1;
  int *dev_b1;
  int *dev_c1;

  // start the timers
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  // initialize the stream
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream0));
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));

  // allocate the memory on the GPU
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_a0, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_b0, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_c0, N * sizeof(int)));

  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_a1, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_b1, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_c1, N * sizeof(int)));

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
  for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
    // copy the locked memory to the device, async
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dev_a0, host_a + i,
                                     N * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream0));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dev_b0, host_b + i,
                                     N * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream0));
    // <<<grid dims, block dims, dynamic shared memory size, stream ID>>>
    kernel<<<N / 256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);

    // copy the data from device to locked memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_c + i, dev_c0,
                                     N * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream0));

    // copy the locked memory to the device, async
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dev_a1, host_a + i + N,
                                     N * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dev_b1, host_b + i + N,
                                     N * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream1));

    kernel<<<N / 256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);

    // copy the data from device to locked memory
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_c + i + N, dev_c1,
                                     N * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream1));
  }
  // copy result chunk from locked to full buffer
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream0));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));

  CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));

  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time,
                                        start, stop));
  printf("Time taken:  %3.1f ms\n", elapsed_time);

  // cleanup the streams and memory
  CHECK_CUDA_ERROR(cudaFreeHost(host_a));
  CHECK_CUDA_ERROR(cudaFreeHost(host_b));
  CHECK_CUDA_ERROR(cudaFreeHost(host_c));
  CHECK_CUDA_ERROR(cudaFree(dev_a0));
  CHECK_CUDA_ERROR(cudaFree(dev_b0));
  CHECK_CUDA_ERROR(cudaFree(dev_c0));
  CHECK_CUDA_ERROR(cudaFree(dev_a1));
  CHECK_CUDA_ERROR(cudaFree(dev_b1));
  CHECK_CUDA_ERROR(cudaFree(dev_c1));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream0));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));

  return 0;
}
