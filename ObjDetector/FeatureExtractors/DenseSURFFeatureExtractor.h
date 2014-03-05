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

class DenseSURFFeatureExtractor : public FeatureExtractor
{
    static const Size shapes[3];
    static const int n_cells = 4;
    static const int n_bins = 8;
    static const int step = 4;
    static const int min_cell_edge = 12;

    ifstream filestream;
    Mat sums[n_bins];

    void T2bFilter(const Mat& img_padded, Mat& img_filtered, int bin);
    void ConvertToPatch(int x, int y, Size shape, int cell_edge, Rect& patch);
    void CalcFeature(Rect patch, vector<double>& feature);
    void Normalization(vector<double>& feature);

public:
    string prefix_path;
    Rect win;
    static const int dim = n_bins * n_cells;

    ~DenseSURFFeatureExtractor();
    void LoadFileList(string filename, string prefix_path);
    void IntegralImage(Mat img);
    void ExtractPatches(vector<Rect>& patches);
    void ExtractFeatures(const vector<Rect>& patches, vector<vector<double>>& features_win);
    void ExtractFeatures(const vector<vector<Rect>>& patches, vector<vector<vector<double>>>& features_win);
    bool ExtractNextImageFeatures(const vector<Rect>& patches, vector<vector<double>>& features_img);
    void ProjectPatches(Rect win1, Rect win2, const vector<vector<Rect>>& patches1, vector<vector<Rect>>& patches2);
};

#endif