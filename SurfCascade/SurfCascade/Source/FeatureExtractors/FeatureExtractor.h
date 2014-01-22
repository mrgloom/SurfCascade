#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <opencv2/opencv.hpp>

using cv::Size;

class FeatureExtractor
{
protected:
    static const Size win_size;

public:
    virtual ~FeatureExtractor();
};

#endif