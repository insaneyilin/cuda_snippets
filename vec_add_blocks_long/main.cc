/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include <iostream>
#include <vector>
#include <string>

#include "vec_add.h"

static void PrintVec(int n, int *vec) {
  for (int i = 0; i < n; ++i) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  const int n = 33 * 1024;
  const int num_blocks = 128;
  const int num_threads = 128;
  int *vec1 = nullptr;
  int *vec2 = nullptr;
  int *vec3 = nullptr;

  vec1 = (int*)malloc(n * sizeof(int));
  vec2 = (int*)malloc(n * sizeof(int));
  vec3 = (int*)malloc(n * sizeof(int));

  for (int i = 0; i < n; ++i) {
    vec1[i] = i;
    vec2[i] = 2 * i;
    vec3[i] = 0;
  }

  cudaVecAdd(num_blocks, num_threads, n, vec1, vec2, vec3);

  PrintVec(n, vec1);
  PrintVec(n, vec2);
  PrintVec(n, vec3);

  bool success = true;
  for (int i = 0; i < n; ++i) {
    if (vec1[i] + vec2[i] != vec3[i]) {
      success = false;
      break;
    }
  }
  if (success) {
    printf("We did it!\n");
  }

  free(vec3);
  free(vec2);
  free(vec1);

  return 0;
}

