#include "pcjbf.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace depth_sr {
Classification classify_pixels(const cv::Mat& rgb, const cv::Mat& lr, int factor,
                               float depth_threshold, float color_threshold) {
    CV_Assert(rgb.type() == CV_8UC3 && lr.type() == CV_32FC1);
    cv::Mat valid = lr > 0, validf;
    valid.convertTo(validf, CV_32F, 1.0/255.0);
    cv::Mat n, total, squared, variance, mean, tmp;
    cv::boxFilter(validf, n, CV_32F, cv::Size(5,5), cv::Point(-1,-1), false, cv::BORDER_REFLECT);
    cv::boxFilter(lr.mul(validf), total, CV_32F, cv::Size(5,5), cv::Point(-1,-1), false, cv::BORDER_REFLECT);
    cv::boxFilter(lr.mul(lr).mul(validf), squared, CV_32F, cv::Size(5,5), cv::Point(-1,-1), false, cv::BORDER_REFLECT);
    cv::max(n, 1.0, n);
    cv::divide(total, n, mean);
    cv::divide(squared, n, variance);
    variance -= mean.mul(mean);
    cv::max(variance, 0.0, variance);
    double minv, maxv;
    cv::minMaxLoc(lr, &minv, &maxv, nullptr, nullptr, valid);
    const double range2 = std::max(1.0, (maxv-minv)*(maxv-minv));
    cv::Mat depth_lr = variance / range2 >= depth_threshold;
    Classification result;
    cv::resize(depth_lr, result.depth_edge, rgb.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat gray, high, low, difference;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    cv::dilate(gray, high, cv::Mat::ones(5,5,CV_8U));
    cv::erode(gray, low, cv::Mat::ones(5,5,CV_8U));
    cv::subtract(high, low, difference);
    result.color_edge = difference >= color_threshold * 255.0f;
    result.classes = cv::Mat(rgb.size(), CV_8U);
    for (int y=0; y<rgb.rows; ++y) for (int x=0; x<rgb.cols; ++x) {
        const bool d = result.depth_edge.at<uchar>(y,x) != 0;
        const bool c = result.color_edge.at<uchar>(y,x) != 0;
        result.classes.at<uchar>(y,x) = d ? (c ? 1 : 2) : (c ? 3 : 4);
    }
    return result;
}

cv::Mat pcjbf_refine(const cv::Mat& rgb, const cv::Mat& lr,
                     const cv::Mat& jbu, const Classification& cls, int factor,
                     float sigma_s, float sigma_r) {
    CV_Assert(lr.type()==CV_32FC1 && jbu.type()==CV_32FC1 && jbu.size()==rgb.size());
    cv::Mat gray, output = jbu.clone();
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    for (int y=0; y<output.rows; ++y) for (int x=0; x<output.cols; ++x) {
        const int ly=y/factor, lx=x/factor, category=cls.classes.at<uchar>(y,x);
        const int radius = category==2 ? 0 : 2;
        const int extra = category==2 ? 2 : 3;
        double best = category==1 ? -std::numeric_limits<double>::infinity() :
                                   std::numeric_limits<double>::infinity();
        float value = jbu.at<float>(y,x);
        for (int qy=std::max(0,ly-radius); qy<std::min(lr.rows,ly+extra); ++qy)
            for (int qx=std::max(0,lx-radius); qx<std::min(lr.cols,lx+extra); ++qx) {
                const float candidate = lr.at<float>(qy,qx);
                if (!(candidate>0 && std::isfinite(candidate))) continue;
                double score;
                if (category==1) {
                    const int hy=std::min(output.rows-1,qy*factor+factor/2);
                    const int hx=std::min(output.cols-1,qx*factor+factor/2);
                    const double distance=std::hypot(double(ly-qy),double(lx-qx))/2.0;
                    const double color=std::abs(int(gray.at<uchar>(y,x))-int(gray.at<uchar>(hy,hx)))/255.0;
                    score=std::exp(-distance/(2*sigma_s)-color/(2*sigma_r))*candidate;
                    if (score>best) { best=score; value=candidate; }
                } else {
                    score=std::abs(candidate-jbu.at<float>(y,x));
                    if (score<best) { best=score; value=candidate; }
                }
            }
        output.at<float>(y,x)=value;
    }
    return output;
}

Scores evaluate(const cv::Mat& prediction, const cv::Mat& gt, float bad_threshold) {
    CV_Assert(prediction.type()==CV_32FC1 && gt.type()==CV_32FC1 && prediction.size()==gt.size());
    size_t total=0, bad=0;
    double sum2=0;
    for (int y=0; y<gt.rows; ++y) for (int x=0; x<gt.cols; ++x) {
        const float truth=gt.at<float>(y,x), estimate=prediction.at<float>(y,x);
        if (!(truth>0 && std::isfinite(truth) && std::isfinite(estimate))) continue;
        const double error=std::abs(double(estimate)-truth);
        ++total; bad+=error>bad_threshold; sum2+=error*error;
    }
    if (!total) throw std::runtime_error("No valid GT pixels");
    return {100.0*bad/total, std::sqrt(sum2/total), total};
}
} // namespace depth_sr
