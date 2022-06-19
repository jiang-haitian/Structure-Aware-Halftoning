#ifndef __SAH_HPP__
#define __SAH_HPP__

#include "opencv2/core.hpp"

#include "ostromoukhov.hpp"

class SAH{
    private:
        Ostromoukhov error_diffusioner;

        double w_g_;
        double w_t_;
        double temperature_init_;
        double anneal_factor_;
        double limit_;

        const int win_size = 11;
        const int ksize_half = (win_size - 1) / 2;

        // mse
        cv::Mat kernel_mse;

        cv::Mat mse_ct_blur;
        cv::Mat mse_ht_blur;

        double delta_mse(cv::Point& point, bool isBlack);
        double delta_mse_overlap(cv::Point& white_point, cv::Point& black_point);

        // ssim
        const double k1 = 0.01;
        const double c1 = pow(k1, 2.0);
        const double k2 = 0.03;
        const double c2 = pow(k2, 2.0);
        cv::Mat kernel_ssim;

        cv::Mat ct_;
        cv::Mat ssim_mu_ht;
        cv::Mat ssim_mu_ct;
        cv::Mat ssim_mu_ct_sq;
        cv::Mat ssim_sigma_ct_sq;
        cv::Mat ssim_ht_ct_blur;

        double delta_ssim(cv::Point& point, bool isBlack);
        double delta_ssim_overlap(cv::Point& white_point, cv::Point& black_point);

    public:
        SAH(double w_g=0.98, double temperature_init=0.2, double anneal_factor=0.8, double limit=0.01);
        ~SAH(){}
        cv::Mat error_diffusion(cv::Mat& ct);
        cv::Mat process(cv::Mat& ct);
        cv::Mat process(cv::Mat& ct, cv::Mat& ht_init);
};

#endif
