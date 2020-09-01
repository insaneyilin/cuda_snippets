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
  const int n = 10;
  int vec1[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  int vec2[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  int vec3[10] = { 0 };

  cudaVecAdd(10, vec1, vec2, vec3);

  PrintVec(n, vec1);
  PrintVec(n, vec2);
  PrintVec(n, vec3);

  return 0;
}

