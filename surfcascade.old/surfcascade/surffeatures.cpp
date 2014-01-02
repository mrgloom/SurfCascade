#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "surffeatures.h"
#include "cascadeclassifier.h"

using namespace cv;

CvSURFFeatureParams::CvSURFFeatureParams()
{
    maxCatCount = 0;
    name = SURFF_NAME;
    featSize = CvSURFEvaluator::n_Bins * CvSURFEvaluator::n_Cells;
}

const cv::Size CvSURFEvaluator::shapeSize[3] = { cv::Size(2, 2), cv::Size(1, 4), cv::Size(4, 1) };

void CvSURFEvaluator::init(const CvFeatureParams *_featureParams, int _maxSampleCount, Size _winSize)
{
    CV_Assert(_maxSampleCount > 0);
    for (int bin = 0; bin < n_Bins; bin++)
        sum[bin].create((int)_maxSampleCount, (_winSize.width + 1) * (_winSize.height + 1), CV_32FC1);
    CvFeatureEvaluator::init(_featureParams, _maxSampleCount, _winSize);
}

void CvSURFEvaluator::t2b_filter(const Mat& imgp, Mat& imgd, int bin)
{
    int d;
    for (int y = 1; y < winSize.height + 1; y++) {
        for (int x = 1; x < winSize.width + 1; x++) {
            switch (bin) {
                /*
                 * 0: |dx| - dx
                 * 1: |dx| + dx
                 * 2: |dy| - dy
                 * 3: |dy| + dy
                 * 4: |du| - du
                 * 5: |du| + du
                 * 6: |dv| - dv
                 * 7: |dv| + dv
                 */
            case 0:
            case 1:
                d = -imgp.at<int>(y, x - 1) + imgp.at<int>(y, x + 1);
                break;
            case 2:
            case 3:
                d = -imgp.at<int>(y - 1, x) + imgp.at<int>(y + 1, x);
                break;
            case 4:
            case 5:
                d = -imgp.at<int>(y - 1, x - 1) + imgp.at<int>(y + 1, x + 1);
                break;
            case 6:
            case 7:
                d = -imgp.at<int>(y + 1, x - 1) + imgp.at<int>(y - 1, x + 1);
                break;
            }
            imgd.at<int>(y - 1, x - 1) = abs(d) + d * (bin % 2 ? 1 : -1);
        }
    }
}

void CvSURFEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    CV_DbgAssert( !sum[0].empty() );
    CvFeatureEvaluator::setImage(img, clsLabel, idx);
    Mat imgp(winSize.height + 1, winSize.width + 1, img.type());
    copyMakeBorder(img, imgp, 1, 1, 1, 1, BORDER_REPLICATE);

    Mat imgd(winSize.height, winSize.width, img.type());
    for (int i = 0; i < sizeof(sum); i++) {
        Mat innSum(winSize.height + 1, winSize.width + 1, sum[i].type(), sum[i].ptr<int>((int)idx));
        t2b_filter(imgp, imgd, i);
        integral(imgd, innSum);
    }
}

void CvSURFEvaluator::writeFeatures(FileStorage &fs, const Mat& featureMap) const
{
    _writeFeatures(features, fs, featureMap);
}

void CvSURFEvaluator::generateFeatures()
{
    int offset = winSize.width + 1;
    for (int shape = 0; shape < sizeof(shapeSize); shape++)
    for (int cellEdge = 6; cellEdge <= winSize.width / 2; cellEdge++)
    for (int y = 0; y < winSize.height; y += step)
    for (int x = 0; x < winSize.width; x += step)
        features.push_back(Feature(offset, x, y, shape, cellEdge));
                
    numFeatures = (int)features.size();
}

CvSURFEvaluator::Feature::Feature(int offset, int x, int y, int shape, int cellEdge)
{
    for (int h = 0; h < shapeSize[shape].height; h++)
    for (int w = 0; w < shapeSize[shape].width; w++)
        rect[h * shapeSize[shape].width + w] = cvRect(x, y, cellEdge, cellEdge); //TODO

    for (int i = 0; i < n_Cells; i++) {
        CV_SUM_OFFSETS(fastRect[i].p0, fastRect[i].p1, fastRect[i].p2, fastRect[i].p3, rect[i], offset);
    }
}

void CvSURFEvaluator::Feature::write(FileStorage &fs) const
{
    //fs << CC_RECT << "[:" << rect.x << rect.y << rect.width << rect.height << "]";
}
