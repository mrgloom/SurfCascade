#ifndef SURFCASCADE_H
#define SURFCASCADE_H

#include <opencv2/opencv.hpp>
#include <array>

static const cv::Size shapes[3] = { cv::Size(2, 2), cv::Size(1, 4), cv::Size(4, 1) };
static const int n_bins = 8;
static const int n_cells = 4;
static const int dim = n_bins * n_cells;
static const int step = 4;
static const int min_cell_edge = 3;

void t2b_filter(const cv::Mat& img_padded, cv::Mat& img_filtered, int bin);
void get_feature_rects(int x, int y, cv::Size shape, int cell_edge, cv::Rect rects[]);
void calc_feature_value(const cv::Mat sums[], cv::Rect rects[], std::array<float, dim>& feature);
void normalization(std::array<float, dim>& feature);

//class Feature
//{
//public:
//    Feature(int x, int y, int shape, int cellEdge);
//    float calc(const Mat sum[]);
//
//    cv::Rect rect[n_cells];
//};


#endif