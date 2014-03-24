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

class DenseSURFFeatureExtractor : public FeatureExtractor
{
    static const Size shapes[3];
    static const int n_cells = 4;
    static const int n_bins = 8;
    static const int step = 2;
    static const int min_cell_edge = 4;

    ifstream filestream;
    Mat sums[n_bins];

    void T2bFilter(const Mat& img_padded, Mat& img_filtered, int bin);
    void CalcFeature(const Rect& patch, vector<float>& feature);
    void Normalization(vector<float>& feature);

public:
    string prefix_path;
    Size size;
    static const int dim = n_bins * n_cells;

    ~DenseSURFFeatureExtractor();
    void LoadFileList(string filename, string prefix_path, bool set_size);
    void IntegralImage(Mat img);
    void ExtractPatches(vector<Rect>& patches);
    void ExtractFeatures(const vector<Rect>& patches, vector<vector<float>>& features_win);
    void ExtractFeatures(const vector<vector<Rect>>& patches, vector<vector<vector<float>>>& features_win);
    bool ExtractNextImageFeatures(const vector<Rect>& patches, vector<vector<float>>& features_img);
    bool FillNegSamples(const vector<Rect>& patches, vector<vector<vector<float>>>& features_all, int n_total, CascadeClassifier& cascade_classifier, bool first);
    void ProjectPatches(const Rect win2, const vector<vector<Rect>>& patches1, vector<vector<Rect>>& patches2);
    void ProjectPatches(const Rect win2, const vector<Rect>& patches1, vector<Rect>& patches2);
};

#endif