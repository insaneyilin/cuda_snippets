/* 
 * Copyright 2020 insaneyilin All Rights Reserved.
 * 
 * 
 */

#ifndef COMMON_OCV_IMAGE_H_
#define COMMON_OCV_IMAGE_H_

#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class OCVImage {
 public:
  OCVImage(int w, int h) {
    mat_ = cv::Mat::zeros(w, h, CV_8UC4);
  }

  unsigned char* get_ptr() const {
    return (unsigned char*)mat_.data;
  }

  // number of bytes
  long image_size() const {
    return mat_.cols * mat_.rows * 4;
  }

  char show(const std::string &window_name, int time = 0) {
    cv::imshow(window_name, mat_);
    return cv::waitKey(time);
  }

  cv::Mat mat_;
};

#endif

