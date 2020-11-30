/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include "../common/common.h"
#include "../common/image.h"

#define DIM 1024

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

struct Sphere {
  float r = 0.f;
  float g = 0.f;
  float b = 0.f;

  float x = 0.f;
  float y = 0.f;
  float z = 0.f;

  float radius = 0.f;

  // Given a ray shot from the pixel at (ox, oy), this method computes whether
  // the ray intersects the sphere. If the ray does intersect the sphere, the
  // method computes the distance from the camera where the ray hits the sphere
  // Note: the camera coordinates system is x/y/z <=> right/up/outward,
  // so `dz + z` si ok
  __device__ float hit(float ox, float oy, float *n) {
    float dx = ox - x;
    float dy = oy - y;
    if (dx * dx + dy * dy < radius * radius) {
      float dz = sqrtf(radius * radius - dx * dx - dy * dy);
      *n = dz / sqrtf(radius * radius);
      return dz + z;
    }
    return -INF;
  }
};
#define NUM_SPHERES 100

// use global memory for spheres dataset
__constant__ Sphere g_spheres[NUM_SPHERES];

__global__ void Kernel(unsigned char *ptr) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  float ox = (x - DIM/2);
  float oy = (y - DIM/2);

  float r = 0.f;
  float g = 0.f;
  float b = 0.f;
  float maxz = -INF;
  for (int i = 0; i < NUM_SPHERES; ++i) {
    float n;
    float t = g_spheres[i].hit(ox, oy, &n);
    if (t > maxz) {
      float fscale = n;
      r = g_spheres[i].r * fscale;
      g = g_spheres[i].g * fscale;
      b = g_spheres[i].b * fscale;
      maxz = t;
    }
  }

  ptr[offset * 4 + 0] = (int)(r * 255);
  ptr[offset * 4 + 1] = (int)(g * 255);
  ptr[offset * 4 + 2] = (int)(b * 255);
  ptr[offset * 4 + 3] = 255;
}

int main(int argc, char **argv) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

  Image image(DIM, DIM, 4);
  image.Clear(255);
  unsigned char *dev_bitmap;

  // allocate memory on the GPU for the output bitmap
  const int image_data_size = image.width() * image.height() *
      image.channels();
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_bitmap,
      image_data_size));

  // allocate temp memory for Sphere dataset on the CPU, initialize it, copy to
  // memory on the GPU, then free our temp memory
  Sphere *temp_spheres = (Sphere*)malloc(sizeof(Sphere) * NUM_SPHERES);
  for (int i = 0; i < NUM_SPHERES; ++i) {
    temp_spheres[i].r = rnd(1.0f);
    temp_spheres[i].g = rnd(1.0f);
    temp_spheres[i].b = rnd(1.0f);
    temp_spheres[i].x = rnd(1000.0f) - 500;
    temp_spheres[i].y = rnd(1000.0f) - 500;
    temp_spheres[i].z = rnd(1000.0f) - 500;
    temp_spheres[i].radius = rnd(100.0f) + 20;
  }
  CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_spheres, temp_spheres,
                                      sizeof(Sphere) * NUM_SPHERES));

  free(temp_spheres);

  // generate a bitmap from our sphere data
  dim3 grid_size(DIM/16, DIM/16);
  dim3 block_size(16, 16);
  Kernel<<<grid_size, block_size>>>(dev_bitmap);

  // copy our bitmap back from the GPU for display
  CHECK_CUDA_ERROR(cudaMemcpy(image.mutable_data(), dev_bitmap,
      image_data_size, cudaMemcpyDeviceToHost));

  // get stop time, and display the timing results
  CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime,
                                        start, stop));
  printf("Time to generate:  %3.1f ms\n", elapsedTime);

  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  CHECK_CUDA_ERROR(cudaFree(dev_bitmap));

  image.Write("ray_tracing_const_mem.png");

  return 0;
}
