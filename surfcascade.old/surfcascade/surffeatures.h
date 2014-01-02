#ifndef _OPENCV_SURFFEATURES_H_
#define _OPENCV_SURFFEATURES_H_

#include "traincascade_features.h"

#define SURFF_NAME "surfFeatureParams"
struct CvSURFFeatureParams : CvFeatureParams
{
    CvSURFFeatureParams();

};

class CvSURFEvaluator : public CvFeatureEvaluator
{
public:
    static const unsigned step = 4;
    static const unsigned n_Bins = 8;
    static const unsigned n_Cells = 4;
    static const cv::Size shapeSize[3];
    //static cv::Size shapeSize[3];
    ////CvSURFEvaluator() {
    ////    shapeSize[0] = cv::Size(2, 2);
    //    shapeSize[1] = cv::Size(1, 4);
    //    shapeSize[2] = cv::Size(4, 1);
    //}
    virtual ~CvSURFEvaluator() {}
    virtual void init(const CvFeatureParams *_featureParams,
        int _maxSampleCount, cv::Size _winSize);
    virtual void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    virtual float operator()(int varIdx, int sampleIdx) const;
    virtual void writeFeatures(cv::FileStorage &fs, const cv::Mat& featureMap) const;
protected:
    virtual void generateFeatures();
    virtual void t2b_filter(const cv::Mat& imgp, cv::Mat& imgd, int bin);

    class Feature
    {
    public:
        Feature(int offset, int x, int y, int shape, int cellEdge);
        float calc(const cv::Mat _sum[], size_t y, int featComponent) const;
        void write(cv::FileStorage &fs) const;

        cv::Rect rect[n_Cells];

        struct
        {
            int p0, p1, p2, p3;
        } fastRect[n_Cells];
    };
    std::vector<Feature> features;

    cv::Mat sum[n_Bins]; // 8-bin integral image
};

inline float CvSURFEvaluator::operator()(int varIdx, int sampleIdx) const
{
    int featureIdx = varIdx / (n_Bins * n_Cells);
    int componentIdx = varIdx % (n_Bins * n_Cells);
    return (float)features[featureIdx].calc(sum, sampleIdx, componentIdx);
}

inline float CvSURFEvaluator::Feature::calc(const cv::Mat _sum[], size_t y, int featComponent) const
{
    int binIdx = featComponent % n_Bins;
    int cellIdx = featComponent / n_Bins;
    const float* psum = _sum[binIdx].ptr<float>((int)y);
    return psum[fastRect[cellIdx].p0] - psum[fastRect[cellIdx].p1] - psum[fastRect[cellIdx].p2] + psum[fastRect[cellIdx].p3];
}

#endif
