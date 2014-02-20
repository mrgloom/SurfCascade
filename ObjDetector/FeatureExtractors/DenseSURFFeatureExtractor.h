#ifndef DENSESURFFEATUREEXTRACTOR_H
#define DENSESURFFEATUREEXTRACTOR_H

#include "FeatureExtractors/FeatureExtractor.h"
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>

using cv::Size;
using cv::Mat;
using cv::Rect;
using std::array;
using std::string;
using std::vector;

class DenseSURFFeatureExtractor : public FeatureExtractor
{
    static const Size shapes[3];
    static const int n_cells = 4;
    static const int step = 2;
    static const int min_cell_edge = 3;

    void T2bFilter(const Mat& img_padded, Mat& img_filtered, int bin);
    void GetFeatureRects(int x, int y, Size shape, int cell_edge, Rect rects[]);
    void CalcFeatureValue(const Mat sums[], Rect rects[], vector<double>& feature);
    void Normalization(vector<double>& feature);

public:
    static const int n_bins = 8;
    static const int dim = n_bins * n_cells;

    ~DenseSURFFeatureExtractor();
    void IntegralImage(string filename, Mat sums[]);
    void ExtractFeatures(const Mat sums[], const Rect& win, vector<vector<double>>& features_win);
};

#endif