/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include "../common/common.h"

int main(int argc, char **argv) {
  cudaDeviceProp prop;
  int dev_id = -1;

  CHECK_CUDA_ERROR(cudaGetDevice(&dev_id));
  printf("ID of current CUDA device:  %d\n", dev_id);

  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 3;
  CHECK_CUDA_ERROR(cudaChooseDevice(&dev_id, &prop));
  printf("ID of CUDA device closest to revision 1.3:  %d\n", dev_id);

  CHECK_CUDA_ERROR(cudaSetDevice(dev_id));

  return 0;
}

