/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#include "../common/common.h"

#define SIZE (100*1024*1024)

int main(int argc, char **argv) {
  // allocate buffer with random values
  unsigned char *buffer =
      (unsigned char*)big_random_block(SIZE);

  for (int i = 0; i < SIZE; ++i) {
    buffer[i] = buffer[i] % 256;
  }

  clock_t start;
  clock_t stop;
  start = clock();
  unsigned int histo[256];
  for (int i = 0; i < 256; ++i) {
    histo[i] = 0;
  }
  for (int i = 0; i < SIZE; ++i) {
    ++histo[buffer[i]];
  }
  stop = clock();
  float elapsed_time = (float)(stop - start) /
                      (float)CLOCKS_PER_SEC * 1000.0f;

  printf("CPU histogram time: %3.1f ms\n", elapsed_time);

  long histo_count = 0;
  for (int i = 0; i < 256; ++i) {
    histo_count += histo[i];
    printf("%d ", histo[i]);
  }
  printf("\n");
  printf("Histogram Sum: %ld\n", histo_count);

  free(buffer);

  return 0;
}
