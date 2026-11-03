#pragma once
#include <opencv2/opencv.hpp>

namespace depth_sr {

struct Classification {
    cv::Mat depth_edge; // CV_8U, 0/255
    cv::Mat color_edge; // CV_8U, 0/255
    cv::Mat classes;    // CV_8U, 1..4
};

// Semantic transplant of upstream upsample/joint_bilateral_filter at (1,10).
cv::Mat source_jbu(const cv::Mat& rgb, const cv::Mat& lr, int factor);
Classification classify_pixels(const cv::Mat& rgb, const cv::Mat& lr, int factor,
                               float depth_threshold = 0.005f, float color_threshold = 0.10f);
cv::Mat pcjbf_refine(const cv::Mat& rgb, const cv::Mat& lr,
                     const cv::Mat& jbu, const Classification& cls, int factor,
                     float sigma_s = 0.2f, float sigma_r = 0.1f);
struct Scores { double bpr_percent, rmse; size_t valid_count; };
Scores evaluate(const cv::Mat& prediction, const cv::Mat& gt, float bad_threshold = 1.0f);

} // namespace depth_sr
