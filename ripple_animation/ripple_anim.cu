/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include "../common/common.h"
#include "../common/ocv_image.h"

#define DIM 1024
#define PI 3.1415926535897932f

struct DataBlock {
  unsigned char *dev_bitmap;
  OCVImage *bitmap;
};

// clean up memory allocated on the GPU
void CleanUp(DataBlock *d) {
  CHECK_CUDA_ERROR(cudaFree(d->dev_bitmap));
}

__global__ void Kernel(unsigned char *ptr, int ticks) {
  // map from (BlockIdx, threadIdx) to pixel position
  // each thread processes one pixel
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;  // 1d offset

  float fx = x - DIM / 2;
  float fy = y - DIM / 2;
  float d = sqrtf(fx * fx + fy * fy);  // distance to image center
  unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                       cos(d/10.0f - ticks/7.0f) /
                                       (d/10.0f + 1.0f));
  ptr[offset*4 + 0] = grey;
  ptr[offset*4 + 1] = grey;
  ptr[offset*4 + 2] = grey;
  ptr[offset*4 + 3] = 255;
}

int main(int argc, char **argv) {
  DataBlock data;
  OCVImage bitmap(DIM, DIM);
  data.bitmap = &bitmap;

  CHECK_CUDA_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));

  // "size" of the Grid, there are DIM/16 x DIM/16 blocks in the grid
  dim3 blocks(DIM / 16, DIM / 16);
  // "size" of each Block, there are 16 x 16 threads in each block
  dim3 threads(16, 16);

  int ticks = 0;
  bitmap.show("ripple", 30);

  // show ripple animation
  while (1) {
    Kernel<<<blocks, threads>>>(data.dev_bitmap, ticks);

    CHECK_CUDA_ERROR(cudaMemcpy(data.bitmap->get_ptr(),
                                data.dev_bitmap,
                                data.bitmap->image_size(),
                                cudaMemcpyDeviceToHost));

    ++ticks;
    char key = bitmap.show("ripple", 30);
    if (key == 27) {
      break;
    }
  }

  CleanUp(&data);

  return 0;
}
