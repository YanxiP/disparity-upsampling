#include "pcjbf.hpp"
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: pcjbf_experiment RGB_view1.png disp1.png factor output_dir\n";
        return 1;
    }
    try {
        const int factor = std::stoi(argv[3]);
        if (factor != 2 && factor != 4 && factor != 8) throw std::runtime_error("factor must be 2, 4, or 8");
        cv::Mat rgb = cv::imread(argv[1], cv::IMREAD_COLOR);
        cv::Mat raw = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
        if (rgb.empty() || raw.empty() || raw.channels() != 1 || rgb.size() != raw.size())
            throw std::runtime_error("RGB and disparity missing or misaligned");
        const int height = rgb.rows - rgb.rows % 8, width = rgb.cols - rgb.cols % 8;
        rgb = rgb(cv::Rect(0,0,width,height)).clone();
        cv::Mat gt;
        raw(cv::Rect(0,0,width,height)).convertTo(gt, CV_32F);
        cv::Mat lr;
        cv::resize(gt, lr, cv::Size(width/factor,height/factor), 0, 0, cv::INTER_NEAREST);
        const cv::Mat baseline = depth_sr::source_jbu(rgb, lr, factor);
        const auto classes = depth_sr::classify_pixels(rgb, lr, factor);
        const cv::Mat refined = depth_sr::pcjbf_refine(rgb, lr, baseline, classes, factor);
        const auto a = depth_sr::evaluate(baseline, gt);
        const auto b = depth_sr::evaluate(refined, gt);
        std::filesystem::create_directories(argv[4]);
        const std::filesystem::path out = argv[4];
        cv::imwrite((out/"jbu.tiff").string(), baseline);
        cv::imwrite((out/"pcjbf.tiff").string(), refined);
        cv::imwrite((out/"classes.png").string(), classes.classes);
        std::cout << "method,BPR_percent,RMSE,valid_pixels\n";
        std::cout << "JBU," << a.bpr_percent << ',' << a.rmse << ',' << a.valid_count << '\n';
        std::cout << "PCJBF," << b.bpr_percent << ',' << b.rmse << ',' << b.valid_count << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    return 0;
}
