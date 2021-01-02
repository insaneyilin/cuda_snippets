/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include "../common/common.h"
#include "../common/image.h"

#define DIM_W 1920
#define DIM_H 1080

struct Complex {
  __device__ Complex(float rr, float ii) : r(rr), i(ii) {}

  // magnitude squared
  __device__ float MagnitudeSqr() {
    return r * r + i * i;
  }

  __device__ Complex operator*(const Complex &that) {
    return Complex(r * that.r - i * that.i, i * that.r + r * that.i);
  }

  __device__ Complex operator+(const Complex &that) {
    return Complex(r + that.r, i + that.i);
  }

  float r = 0.f;
  float i = 0.f;
};

// return 1 if (x, y) is in the Julia set, else 0
__device__ int IsInJuliaSet(int x, int y) {
  const float scale = 1.5f;
  float jx = scale * (float)(DIM_W / 2 - x) / (DIM_W / 2);
  float jy = scale * (float)(DIM_H / 2 - y) / (DIM_H / 2);

  Complex c(-0.8, 0.156);
  Complex a(jx, jy);

  // Z = Z^2 + C, C is a constant complex number
  for (int i = 0; i < 200; ++i) {
    a = a * a + c;
    // check image (x, y)
    int img_x = (int)((DIM_W / 2) - a.r / scale * (DIM_W / 2));
    int img_y = (int)((DIM_H / 2) - a.i / scale * (DIM_H / 2));
    if (img_x < 0 || img_x >= DIM_W ||
        img_y < 0 || img_y >= DIM_H) {
      return 0;
    }
  }

  return 1;
}

__global__ void Kernel(unsigned char *ptr) {
  // map blockIdx to pixel position
  // each block process one pixel
  int x = blockIdx.x;
  int y = blockIdx.y;

  // gridDim.x gridDim.y <=> DIM_W DIM_H
  int offset = (x + y * gridDim.x) * 4;

  int julia_val = IsInJuliaSet(x, y);
  ptr[offset] = 255 * julia_val;
  ptr[offset + 1] = 0;
  ptr[offset + 2] = 0;
  ptr[offset + 3] = 255;
}

int main(int argc, char **argv) {
  Image image(DIM_W, DIM_H, 4);
  image.Clear(255);

  unsigned char *dev_bitmap = nullptr;
  const int image_data_size = image.width() * image.height() *
      image.channels();
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_bitmap, image_data_size));

  // each block process one pixel
  dim3 grid_size(DIM_W, DIM_H);
  Kernel<<<grid_size, 1>>>(dev_bitmap);

  CHECK_CUDA_ERROR(cudaMemcpy(image.mutable_data(), dev_bitmap,
      image_data_size, cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaFree(dev_bitmap));

  image.Write("julia_gpu.png");

  return 0;
}
