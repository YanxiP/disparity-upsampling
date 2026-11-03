#include "pcjbf.hpp"
#include <cmath>
#include <stdexcept>

namespace depth_sr {
static cv::Mat source_joint_filter(const cv::Mat& guide, const cv::Mat& disp) {
    cv::Mat result(disp.size(), CV_8U);
    for (int y = 0; y < disp.rows; ++y) {
        for (int x = 0; x < disp.cols; ++x) {
            double sum = 0, norm = 0;
            for (int dy = -2; dy <= 2; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= disp.rows) continue;
                for (int dx = -2; dx <= 2; ++dx) {
                    const int xx = x + dx;
                    if (xx < 0 || xx >= disp.cols) continue;
                    const int difference = int(guide.at<uchar>(y,x)) - int(guide.at<uchar>(yy,xx));
                    const double weight = std::exp(-(dx*dx+dy*dy)/2.0 - difference*difference/200.0);
                    norm += weight;
                    sum += weight * disp.at<uchar>(yy,xx);
                }
            }
            result.at<uchar>(y,x) = static_cast<uchar>(std::round(sum/norm));
        }
    }
    return result;
}

cv::Mat source_jbu(const cv::Mat& rgb, const cv::Mat& lr, int factor) {
    if (rgb.type() != CV_8UC3 || lr.type() != CV_32FC1 || (factor != 2 && factor != 4 && factor != 8))
        throw std::invalid_argument("Expected BGR uint8, LR float32 disparity, factor 2/4/8");
    cv::Mat guide, current;
    cv::cvtColor(rgb, guide, cv::COLOR_BGR2GRAY);
    lr.convertTo(current, CV_8U); // original repository's 8-bit input contract
    const int stages = static_cast<int>(std::floor(std::log2(factor)));
    for (int i = 1; i < stages - 1; ++i) {
        cv::resize(current, current, cv::Size(), 2.0, 2.0);
        cv::Mat intermediate;
        cv::resize(guide, intermediate, current.size());
        current = source_joint_filter(intermediate, current);
    }
    cv::resize(current, current, guide.size());
    current = source_joint_filter(guide, current);
    cv::Mat result;
    current.convertTo(result, CV_32F);
    return result;
}
} // namespace depth_sr
