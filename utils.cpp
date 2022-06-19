#include <cmath>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "utils.hpp"

cv::Mat get_gaussian_kernel(int win_size, double sigma){
    std::vector<double> kernel1d(win_size);
    int ksize_half = (win_size - 1) / 2;
    double sum = 0.;
    for(int i=0; i<win_size; i++){
        double x = (double)(i - ksize_half);
        double pdf = exp(-0.5 * pow(x / sigma, 2.));
        kernel1d[i] = pdf;
        sum += pdf;
    }
    for(int i=0; i<win_size; i++){
        kernel1d[i] /= sum;
    }

    cv::Mat kernel(win_size, win_size, CV_64FC1);
    for(int h=0; h<win_size; h++){
        for(int w=0; w<win_size; w++){
            kernel.ptr<double>(h)[w] = kernel1d[h] * kernel1d[w];
        }
    }
    return kernel;
}

cv::Mat conv2d(cv::Mat& input, cv::Mat& weight, bool padding){
    cv::Mat output;
    cv::filter2D(input, output, -1, weight, cv::Point(-1, -1), 0., cv::BORDER_CONSTANT);
    if (!padding){
        int half_k_h = (weight.rows - 1) / 2;
        int half_k_w = (weight.cols - 1) / 2;
        output = output(cv::Rect(half_k_w, half_k_h, output.cols-2*half_k_w, output.rows-2*half_k_h));
    }
    return output;
}

cv::Mat pad_zero(cv::Mat& img, int border){
    cv::Mat img_pad;
    cv::copyMakeBorder(img, img_pad, border, border, border, border, cv::BORDER_CONSTANT, 0.);
    return img_pad;
}

cv::Mat deconv2d(cv::Mat& input, cv::Mat& weight){
    int h = input.rows;
    int w = input.cols;
    int k_h = weight.rows;
    int k_w = weight.cols;
    int half_k_h = (k_h - 1) / 2;
    int half_k_w = (k_w - 1) / 2;

    cv::Mat output;
    cv::copyMakeBorder(input, output, 2*half_k_h, 2*half_k_h, 2*half_k_w, 2*half_k_w, cv::BORDER_CONSTANT, 0.0);
    cv::filter2D(output, output, -1, weight, cv::Point(-1, -1), 0., cv::BORDER_CONSTANT);
    output = output(cv::Rect(half_k_w, half_k_h, w+2*half_k_w, h+2*half_k_h));
    return output;
}

cv::Mat mse_loss(cv::Mat& img1, cv::Mat& img2){
    cv::Mat mse_map = img1 - img2;
    mse_map = mse_map.mul(mse_map);
    return mse_map;
}

double mse_func(cv::Mat& img1, cv::Mat& img2, int win_size){
    cv::Mat img1_, img2_; 
    if (img1.type() == CV_8U)
        img1.convertTo(img1_, CV_64F, 1./255.);
    else
        img1_ = img1.clone();

    if (img2.type() == CV_8U)
        img2.convertTo(img2_, CV_64F, 1./255.);
    else
        img1_ = img2.clone();

    cv::Mat kernel_mse = get_gaussian_kernel(win_size, 2.);
    img1_ = conv2d(img1_, kernel_mse, false);
    img2_ = conv2d(img2_, kernel_mse, false);

    cv::Mat error = mse_loss(img1_, img2_);
    return cv::mean(error)[0];
}

double ssim_func(cv::Mat& img1, cv::Mat& img2){
    cv::Mat img1_, img2_; 
    if (img1.type() == CV_8U)
        img1.convertTo(img1_, CV_64F, 1./255.);
    else
        img1_ = img1.clone();

    if (img2.type() == CV_8U)
        img2.convertTo(img2_, CV_64F, 1./255.);
    else
        img1_ = img2.clone();

    int win_size = 11;
    double k1 = 0.01;
    double k2 = 0.03;
    double c1 = pow(k1, 2.);
    double c2 = pow(k2, 2.);

    int ksize_half = (win_size - 1) / 2;

    cv::Mat kernel_ssim = get_gaussian_kernel(win_size, 1.5);

    cv::Mat mu1 = conv2d(img1_, kernel_ssim, false);
    cv::Mat mu2 = conv2d(img2_, kernel_ssim, false);

    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat img1_sq = img1_.mul(img1_);
    cv::Mat img2_sq = img2_.mul(img2_);
    cv::Mat img1_img2 = img1_.mul(img2_);

    cv::Mat sigma1_sq = conv2d(img1_sq, kernel_ssim, false);
    cv::Mat sigma2_sq = conv2d(img2_sq, kernel_ssim, false);
    sigma1_sq -= mu1_sq;
    sigma2_sq -= mu2_sq;

    cv::Mat sigma12 = conv2d(img1_img2, kernel_ssim, false);
    sigma12 -= mu1_mu2;

    cv::Mat cs_map, ssim_map;
    cv::divide(2.*sigma12+c2, sigma1_sq+sigma2_sq+c2, cs_map);
    cv::divide(2.*mu1_mu2+c1, mu1_sq+mu2_sq+c1, ssim_map);
    ssim_map = ssim_map.mul(cs_map);

    return cv::mean(ssim_map)[0];
}
