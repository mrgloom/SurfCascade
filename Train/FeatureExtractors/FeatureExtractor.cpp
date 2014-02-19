#include "FeatureExtractors/FeatureExtractor.h"
#include <opencv2/opencv.hpp>

using cv::Size;

const Size FeatureExtractor::win_size = { 19, 19 };

FeatureExtractor::~FeatureExtractor()
{

}
