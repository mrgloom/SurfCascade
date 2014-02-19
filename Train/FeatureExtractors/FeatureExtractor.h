#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <opencv2/opencv.hpp>

using cv::Size;

class FeatureExtractor
{
public:
    virtual ~FeatureExtractor();
};

#endif