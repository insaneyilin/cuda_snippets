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
  Complex(float rr, float ii) : r(rr), i(ii) {}

  // magnitude squared
  float MagnitudeSqr() {
    return r * r + i * i;
  }

  Complex operator*(const Complex &that) {
    return Complex(r * that.r - i * that.i, i * that.r + r * that.i);
  }

  Complex operator+(const Complex &that) {
    return Complex(r + that.r, i + that.i);
  }

  float r = 0.f;
  float i = 0.f;
};

// return 1 if (x, y) is in the Julia set, else 0
int IsInJuliaSet(int x, int y) {
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

void Kernel(unsigned char *ptr) {
  for (int y = 0; y < DIM_H; ++y) {
    for (int x = 0; x < DIM_W; ++x) {
      int offset = (x + y * DIM_W) * 4;
      int julia_val = IsInJuliaSet(x, y);
      ptr[offset] = 255 * julia_val;
      ptr[offset + 1] = 0;
      ptr[offset + 2] = 0;
      ptr[offset + 3] = 255;
    }
  }
}

int main(int argc, char **argv) {
  Image image(DIM_W, DIM_H, 4);
  image.Clear(255);
  unsigned char *ptr = image.mutable_data();
  Kernel(ptr);
  image.Write("julia_cpu.png");

  return 0;
}
