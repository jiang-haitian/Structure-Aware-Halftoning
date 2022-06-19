#include <sys/time.h>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "sah.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]){
    if (argc != 2 && argc != 3){ 
        std::cout << "Usage:" << std::endl
            << "error_diffusion INPUT_IMG [OUTPUT_IMG]" << std::endl;
        return -1; 
    }   

    double w_g = 0.99, temperature_init=0.2, anneal_factor=0.8, limit=0.01;
    SAH saher(w_g, temperature_init, anneal_factor, limit);

    cv::Mat ct = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

#if 1
    cv::Mat ht = saher.process(ct);
#else
    struct timeval start, end;
    gettimeofday(&start, NULL);
    cv::Mat ht_init = saher.error_diffusion(ct);
    cv::Mat ht = saher.process(ct, ht_init);
    gettimeofday(&end, NULL);
    long time = end.tv_sec - start.tv_sec;
    std::cout << "time = " << time << "s" << std::endl;

    std::cout << "mse before = " << mse_func(ct, ht_init) << std::endl;
    std::cout << "ssim before = " << ssim_func(ct, ht_init) << std::endl;
    std::cout << "mse after = " << mse_func(ct, ht) << std::endl;
    std::cout << "ssim after = " << ssim_func(ct, ht) << std::endl;
#endif

    if (argc == 3){ 
        cv::imwrite(argv[2], ht);
    }else{
        cv::imshow("ht", ht); 
        cv::waitKey(0);
    }  
    return 0;
}
