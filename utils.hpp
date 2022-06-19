#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "opencv2/core.hpp"

cv::Mat get_gaussian_kernel(int win_size, double sigma);
cv::Mat conv2d(cv::Mat& input, cv::Mat& weight, bool padding=false);
cv::Mat pad_zero(cv::Mat& img, int border);
cv::Mat deconv2d(cv::Mat& input, cv::Mat& weight);
cv::Mat mse_loss(cv::Mat& img1, cv::Mat& img2);
double mse_func(cv::Mat& img1, cv::Mat& img2, int win_size=11);
double ssim_func(cv::Mat& img1, cv::Mat& img2);

#endif
