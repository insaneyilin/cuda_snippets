/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#ifndef VEC_ADD_VEC_ADD_H_
#define VEC_ADD_VEC_ADD_H_

#include <stdio.h>
#include <cuda_runtime.h>

#include "../common/common.h"

void cudaVecAdd(int vec_len, int *in_v1, int *in_v2, int *out_v);

#endif  // VEC_ADD_VEC_ADD_H_

