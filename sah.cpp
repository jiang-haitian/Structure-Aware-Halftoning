#include <cmath>
#include <iostream>
#include <algorithm>
#include <experimental/random>
#include "opencv2/imgproc.hpp"

#include "sah.hpp"
#include "utils.hpp"

using std::experimental::randint;

SAH::SAH(double w_g, double temperature_init, double anneal_factor, double limit)
    : w_g_(w_g), w_t_(1.0-w_g), temperature_init_(temperature_init), anneal_factor_(anneal_factor), limit_(limit){
        kernel_mse = get_gaussian_kernel(win_size, 2.);
        kernel_ssim = get_gaussian_kernel(win_size, 1.5);

        std::experimental::reseed(1);
        //std::experimental::reseed();
    }

double SAH::delta_mse(cv::Point& point, bool isBlack){
    cv::Mat ct_win(mse_ct_blur, cv::Rect(point.x, point.y, win_size, win_size));
    cv::Mat ht_win(mse_ht_blur, cv::Rect(point.x, point.y, win_size, win_size));
    double mse_local_sum = cv::sum(mse_loss(ct_win, ht_win))[0];

    cv::Mat ht_win_new = ht_win.clone();
    if (isBlack)
        ht_win_new += kernel_mse;
    else
        ht_win_new -= kernel_mse;
    double mse_local_sum_new = cv::sum(mse_loss(ct_win, ht_win_new))[0];

    return (mse_local_sum_new - mse_local_sum) / mse_ht_blur.total();
}

double SAH::delta_mse_overlap(cv::Point& white_point, cv::Point& black_point){
    int x_min = std::min(white_point.x, black_point.x);
    int x_dist = abs(white_point.x - black_point.x);
    int y_min = std::min(white_point.y, black_point.y);
    int y_dist = abs(white_point.y - black_point.y);
    cv::Rect roi(x_min, y_min, win_size+x_dist, win_size+y_dist);

    cv::Mat ct_win(mse_ct_blur, roi);
    cv::Mat ht_win(mse_ht_blur, roi);
    double mse_local_sum = cv::sum(mse_loss(ct_win, ht_win))[0];

    cv::Mat ht_win_new = ht_win.clone();
    cv::Rect white_roi(white_point.x-x_min, white_point.y-y_min, win_size, win_size);
    cv::Rect black_roi(black_point.x-x_min, black_point.y-y_min, win_size, win_size);

    ht_win_new(white_roi) -= kernel_mse;
    ht_win_new(black_roi) += kernel_mse;
    double mse_local_sum_new = cv::sum(mse_loss(ct_win, ht_win_new))[0];

    return (mse_local_sum_new - mse_local_sum) / mse_ht_blur.total();
}

double SAH::delta_ssim(cv::Point& point, bool isBlack){
    cv::Rect roi(point.x, point.y, win_size, win_size);

    cv::Mat mu_ht = ssim_mu_ht(roi);
    cv::Mat mu_ct = ssim_mu_ct(roi);
    cv::Mat ht_ct_blur = ssim_ht_ct_blur(roi);
    cv::Mat mu_ct_sq = ssim_mu_ct_sq(roi);
    cv::Mat sigma_ct_sq = ssim_sigma_ct_sq(roi);

    cv::Mat mu_ht_sq = mu_ht.mul(mu_ht);
    cv::Mat mu_ht_mu_ct = mu_ht.mul(mu_ct);
    cv::Mat ht_sq_blur = mu_ht;
    cv::Mat sigma_ht_sq = ht_sq_blur - mu_ht_sq;
    cv::Mat sigma_ht_ct = ht_ct_blur - mu_ht_mu_ct;

    cv::Mat cs_map, ssim_map;
    cv::divide(2.*sigma_ht_ct+c2, sigma_ht_sq+sigma_ct_sq+c2, cs_map);
    cv::divide(2.*mu_ht_mu_ct+c1, mu_ht_sq+mu_ct_sq+c1, ssim_map);
    ssim_map = ssim_map.mul(cs_map);

    double ssim_local_sum = cv::sum(ssim_map)[0];

    cv::Mat mu_ht_new, ht_ct_blur_new;
    double ct_value = ct_.at<double>(point);
    if (isBlack){
        mu_ht_new = mu_ht + kernel_ssim;
        ht_ct_blur_new = ht_ct_blur + ct_value * kernel_ssim;
    }else{
        mu_ht_new = mu_ht - kernel_ssim;
        ht_ct_blur_new = ht_ct_blur - ct_value * kernel_ssim;
    }
    mu_ht_sq = mu_ht_new.mul(mu_ht_new);
    mu_ht_mu_ct = mu_ht_new.mul(mu_ct);
    ht_sq_blur = mu_ht_new;
    sigma_ht_sq = ht_sq_blur - mu_ht_sq;
    sigma_ht_ct = ht_ct_blur_new - mu_ht_mu_ct;

    cv::divide(2.*sigma_ht_ct+c2, sigma_ht_sq+sigma_ct_sq+c2, cs_map);
    cv::divide(2.*mu_ht_mu_ct+c1, mu_ht_sq+mu_ct_sq+c1, ssim_map);
    ssim_map = ssim_map.mul(cs_map);

    double ssim_local_sum_new = cv::sum(ssim_map)[0];

    return (ssim_local_sum_new - ssim_local_sum) / ssim_mu_ht.total();
}

double SAH::delta_ssim_overlap(cv::Point& white_point, cv::Point& black_point){
    int x_min = std::min(white_point.x, black_point.x);
    int x_dist = abs(white_point.x - black_point.x);
    int y_min = std::min(white_point.y, black_point.y);
    int y_dist = abs(white_point.y - black_point.y);
    cv::Rect roi(x_min, y_min, win_size+x_dist, win_size+y_dist);

    cv::Mat mu_ht = ssim_mu_ht(roi);
    cv::Mat mu_ct = ssim_mu_ct(roi);
    cv::Mat ht_ct_blur = ssim_ht_ct_blur(roi);
    cv::Mat mu_ct_sq = ssim_mu_ct_sq(roi);
    cv::Mat sigma_ct_sq = ssim_sigma_ct_sq(roi);

    cv::Mat mu_ht_sq = mu_ht.mul(mu_ht);
    cv::Mat mu_ht_mu_ct = mu_ht.mul(mu_ct);
    cv::Mat ht_sq_blur = mu_ht;
    cv::Mat sigma_ht_sq = ht_sq_blur - mu_ht_sq;
    cv::Mat sigma_ht_ct = ht_ct_blur - mu_ht_mu_ct;

    cv::Mat cs_map, ssim_map;
    cv::divide(2.*sigma_ht_ct+c2, sigma_ht_sq+sigma_ct_sq+c2, cs_map);
    cv::divide(2.*mu_ht_mu_ct+c1, mu_ht_sq+mu_ct_sq+c1, ssim_map);
    ssim_map = ssim_map.mul(cs_map);

    double ssim_local_sum = cv::sum(ssim_map)[0];

    cv::Rect white_roi(white_point.x-x_min, white_point.y-y_min, win_size, win_size);
    cv::Rect black_roi(black_point.x-x_min, black_point.y-y_min, win_size, win_size);

    cv::Mat mu_ht_new = mu_ht.clone();
    cv::Mat ht_ct_blur_new = ht_ct_blur.clone();
    mu_ht_new(black_roi) += kernel_ssim;
    mu_ht_new(white_roi) -= kernel_ssim;
    ht_ct_blur_new(black_roi) += ct_.at<double>(black_point) * kernel_ssim;
    ht_ct_blur_new(white_roi) -= ct_.at<double>(white_point) * kernel_ssim;

    mu_ht_sq = mu_ht_new.mul(mu_ht_new);
    mu_ht_mu_ct = mu_ht_new.mul(mu_ct);
    ht_sq_blur = mu_ht_new;
    sigma_ht_sq = ht_sq_blur - mu_ht_sq;
    sigma_ht_ct = ht_ct_blur_new - mu_ht_mu_ct;

    cv::divide(2.*sigma_ht_ct+c2, sigma_ht_sq+sigma_ct_sq+c2, cs_map);
    cv::divide(2.*mu_ht_mu_ct+c1, mu_ht_sq+mu_ct_sq+c1, ssim_map);
    ssim_map = ssim_map.mul(cs_map);

    double ssim_local_sum_new = cv::sum(ssim_map)[0];
    return (ssim_local_sum_new - ssim_local_sum) / ssim_mu_ht.total();
}

cv::Mat SAH::process(cv::Mat& ct, cv::Mat& ht_init){
    cv::Mat ht;
    ht_init.convertTo(ht, CV_64FC1, 1.0/255.0);
    ct.convertTo(ct, CV_64FC1, 1.0/255.0);

    ct_ = ct;

    std::vector<cv::Point> black_points, white_points;
    for (int i=0; i<ht.rows; i++){
        double* row_ptr = ht.ptr<double>(i); 
        for (int j=0; j<ht.cols; j++){
            if (row_ptr[j] > 0.0)
                white_points.push_back({j, i});
            else
                black_points.push_back({j, i});
        }
    }

    /* mse init */
    mse_ht_blur = deconv2d(ht, kernel_mse);
    mse_ct_blur = deconv2d(ct, kernel_mse);

    /* ssim init */
    ssim_mu_ht = deconv2d(ht, kernel_ssim);
    ssim_mu_ct = deconv2d(ct, kernel_ssim);

    ssim_mu_ct_sq = ssim_mu_ct.mul(ssim_mu_ct);
    cv::Mat ssim_ct_sq = ct.mul(ct);
    cv::Mat ssim_ht_ct = ht.mul(ct);
    cv::Mat ssim_ct_sq_blur = deconv2d(ssim_ct_sq, kernel_ssim);
    ssim_sigma_ct_sq = ssim_ct_sq_blur - ssim_mu_ct_sq;
    ssim_ht_ct_blur = deconv2d(ssim_ht_ct, kernel_ssim);

#if 0
    /* full ssim evaluate */
    cv::Mat ssim_mu_ht_sq = ssim_mu_ht.mul(ssim_mu_ht);
    //cv::Mat ssim_mu_ct_sq = ssim_mu_ct.mul(ssim_mu_ct);
    cv::Mat ssim_mu_ht_mu_ct = ssim_mu_ht.mul(ssim_mu_ct);
    cv::Mat ssim_ht_sq = ht.mul(ht);
    //cv::Mat ssim_ct_sq = ct.mul(ct);
    //cv::Mat ssim_ht_ct = ht.mul(ct);
    cv::Mat ssim_ht_sq_blur = deconv2d(ssim_ht_sq, kernel_ssim);
    //cv::Mat ssim_ct_sq_blur = deconv2d(ssim_ct_sq, kernel_ssim);
    cv::Mat ssim_sigma_ht_sq = ssim_ht_sq_blur - ssim_mu_ht_sq;
    //cv::Mat ssim_sigma_ct_sq = ssim_ct_sq_blur - ssim_mu_ct_sq;
    //cv::Mat ssim_ht_ct_blur = deconv2d(ssim_ht_ct, kernel_ssim);
    cv::Mat ssim_sigma_ht_ct = ssim_ht_ct_blur - ssim_mu_ht_mu_ct;

    cv::Mat cs_map, ssim_map;
    cv::divide(2.*ssim_sigma_ht_ct+c2, ssim_sigma_ht_sq+ssim_sigma_ct_sq+c2, cs_map);
    cv::divide(2.*ssim_mu_ht_mu_ct+c1, ssim_mu_ht_sq+ssim_mu_ct_sq+c1, ssim_map);
    ssim_map = ssim_map.mul(cs_map);
    std::cout << "ssim = ", cv::mean(ssim_map)[0] << std::endl;
#endif

    ssim_ct_sq.release();
    ssim_ht_ct.release();
    ssim_ct_sq_blur.release();

    /* optimization */
    double temperature = temperature_init_;
    while(temperature > limit_){
        //std::cout << "temperature = " << temperature << std::endl;

        for (int k=0; k<ht.total(); k++){
            int white_index = randint<int>(0, white_points.size()-1);
            int black_index = randint<int>(0, black_points.size()-1);
            cv::Point& white_point = white_points[white_index];
            cv::Point& black_point = black_points[black_index];

            double d_mse, d_ssim;
            if (abs(white_point.x-black_point.x)<win_size &&
                    abs(white_point.y-black_point.y)<win_size){
                d_mse = delta_mse_overlap(white_point, black_point);
                d_ssim = delta_ssim_overlap(white_point, black_point);
            }else{
                d_mse = delta_mse(white_point, false) + delta_mse(black_point, true);
                d_ssim = delta_ssim(white_point, false) + delta_ssim(black_point, true);
            }
            double d_e = w_g_ * d_mse - w_t_ * d_ssim;

            // accept
            //if (d_e < 0 || rand() < (int)((double)RAND_MAX * exp(-d_e / temperature)))
            if (d_e < 0)
            {
                cv::Rect white_roi(white_point.x, white_point.y, win_size, win_size);
                cv::Rect black_roi(black_point.x, black_point.y, win_size, win_size);

                // mse
                mse_ht_blur(white_roi) -= kernel_mse;
                mse_ht_blur(black_roi) += kernel_mse;
                // ssim
                ssim_mu_ht(white_roi) -= kernel_ssim;
                ssim_mu_ht(black_roi) += kernel_ssim;
                ssim_ht_ct_blur(white_roi) -= ct_.at<double>(white_point) * kernel_ssim;
                ssim_ht_ct_blur(black_roi) += ct_.at<double>(black_point) * kernel_ssim;

                std::swap<cv::Point>(white_point, black_point);
            }
        }

        temperature *= anneal_factor_;
    }

    ht = cv::Mat::zeros(ht.size(), CV_8UC1);
    for (auto& point : white_points){
        ht.at<uint8_t>(point) = 255;
    }
    return ht;
}

cv::Mat SAH::error_diffusion(cv::Mat& ct){
    cv::Mat ht = error_diffusioner.process(ct);
    return ht;
}

cv::Mat SAH::process(cv::Mat& ct){
    cv::Mat ht_init = error_diffusioner.process(ct);
    return process(ct, ht_init);
}
