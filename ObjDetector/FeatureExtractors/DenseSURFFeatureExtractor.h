#ifndef DENSESURFFEATUREEXTRACTOR_H
#define DENSESURFFEATUREEXTRACTOR_H

#include "FeatureExtractors/FeatureExtractor.h"
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <iostream>
#include <fstream>

using cv::Size;
using cv::Mat;
using cv::Rect;
using std::array;
using std::string;
using std::vector;
using std::ifstream;

class CascadeClassifier;

struct F256Dat
{
    __m128 xmm_f1;
    __m128 xmm_f2;
};

typedef cv::Vec<float, 8> Vec8f;

class DenseSURFFeatureExtractor : public FeatureExtractor
{
    static const Size shapes[3];
    static const int n_cells = 4;
    static const int n_bins = 8;
    static const int step = 4;
    static const int min_cell_edge = 6;
    float theta = 2 / sqrt(float(dim));

    Mat summat;
    F256Dat** sumtab;

    void T2bFilter(const Mat& img, uchar *grad);
    void Normalize(vector<float>& feature);

public:
    vector<string> imgnames;
    string prefix_path;
    Size size;
    static const int dim = n_bins * n_cells;

    ~DenseSURFFeatureExtractor();
    void LoadFileList(string filename, string prefix_path, bool set_size);
    void IntegralImage(Mat img);
    void ExtractPatches(vector<Rect>& patches);
    void CalcFeature(const Rect& patch, vector<float>& feature);
    void ExtractFeatures(const vector<Rect>& patches, vector<vector<float>>& features_win);
    void ExtractFeatures(const vector<vector<Rect>>& patches, vector<vector<vector<float>>>& features_win);
    bool ExtractNextImageFeatures(const vector<Rect>& patches, vector<vector<float>>& features_img);
    bool FillNegSamples(const vector<Rect>& patches, vector<vector<vector<float>>>& features_all, int n_total, CascadeClassifier& cascade_classifier, bool first);
    void ProjectPatches(const Rect win2, const vector<vector<Rect>>& patches1, vector<vector<Rect>>& patches2);
    void ProjectPatches(const Rect win2, const vector<Rect>& patches1, vector<Rect>& patches2);
};

#endif