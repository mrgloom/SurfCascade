#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include <vector>
#include <string>
#include <array>
#include <numeric>

using std::cout;
using std::endl;
using std::inner_product;
using std::vector;
using std::string;
using std::array;
using cv::Mat;
using cv::Size;
using cv::Rect;

const Size DenseSURFFeatureExtractor::shapes[3] = { Size(2, 2), Size(1, 4), Size(4, 1) };

DenseSURFFeatureExtractor::~DenseSURFFeatureExtractor()
{
}

void DenseSURFFeatureExtractor::IntegralImage(string filename, Mat sums[])
{
    Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    Mat img_padded;
    Mat img_filtered(img.rows, img.cols, CV_8UC1);

    /* calculate integral image */
    copyMakeBorder(img, img_padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    for (int bin = 0; bin < n_bins; bin++) {
        T2bFilter(img_padded, img_filtered, bin);
        sums[bin].create(img.rows + 1, img.cols + 1, CV_32SC1);
        integral(img_filtered, sums[bin]);
    }
}

void DenseSURFFeatureExtractor::ExtractFeatures(const Mat sums[], const Rect& win, vector<vector<double>>& features_win)
{
    /* compute features */
    Rect rects[n_cells];

    for (int j = 0; j < sizeof(shapes) / sizeof(shapes[0]); j++) {
        Size shape = shapes[j];

        for (int cell_edge = min_cell_edge; cell_edge <= win.width / 2; cell_edge++) {
            for (int y = win.y; y + shape.height * cell_edge <= win.y + win.height; y += step)
            for (int x = win.x; x + shape.width * cell_edge <= win.x + win.width; x += step) {
                GetFeatureRects(x, y, shape, cell_edge, rects);
                vector<double> feature;
                CalcFeatureValue(sums, rects, feature);
                features_win.push_back(feature);
            }
        }
    }
}

void DenseSURFFeatureExtractor::T2bFilter(const Mat& img_padded, Mat& img_filtered, int bin)
{
    int d;
    for (int y = 1; y < img_filtered.rows + 1; y++) {
        for (int x = 1; x < img_filtered.cols + 1; x++) {
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
                d = -img_padded.at<uchar>(y, x - 1) + img_padded.at<uchar>(y, x + 1);
                break;
            case 2:
            case 3:
                d = -img_padded.at<uchar>(y - 1, x) + img_padded.at<uchar>(y + 1, x);
                break;
            case 4:
            case 5:
                d = -img_padded.at<uchar>(y - 1, x - 1) + img_padded.at<uchar>(y + 1, x + 1);
                break;
            case 6:
            case 7:
                d = -img_padded.at<uchar>(y + 1, x - 1) + img_padded.at<uchar>(y - 1, x + 1);
                break;
            }
            img_filtered.at<uchar>(y - 1, x - 1) = abs(d) + d * (bin % 2 ? 1 : -1);
        }
    }
}

void DenseSURFFeatureExtractor::GetFeatureRects(int x, int y, Size shape, int cell_edge, Rect rects[])
{
    for (int h = 0; h < shape.height; h++)
    for (int w = 0; w < shape.width; w++) {
        rects[h * shape.width + w] = Rect(x + w * cell_edge, y + h * cell_edge, cell_edge, cell_edge);
    }
}

void DenseSURFFeatureExtractor::CalcFeatureValue(const Mat sums[], Rect rects[], vector<double>& feature)
{
    /* calculate feature value using integral image*/
    int s0, s1, s2, s3, s;

    feature.resize(dim);

    for (int i = 0; i < n_cells; i++) {
        for (int j = 0; j < n_bins; j++) {
            Mat sum = sums[j];
            s0 = sum.at<int>(rects[i].y, rects[i].x);
            s1 = sum.at<int>(rects[i].y, rects[i].x + rects[i].width);
            s2 = sum.at<int>(rects[i].y + rects[i].height, rects[i].x);
            s3 = sum.at<int>(rects[i].y + rects[i].height, rects[i].x + rects[i].width);
            s = s3 - s2 - s1 + s0;
            feature[i * n_bins + j] = s;
        }
    }

    /* normalization */
    Normalization(feature);
}

void DenseSURFFeatureExtractor::Normalization(vector<double>& feature) {
    double norm;
    norm = sqrt(inner_product(feature.begin(), feature.end(), feature.begin(), 0.0));
    for (int i = 0; i < feature.size(); i++)
        feature[i] /= norm;

    double theta = 2 / sqrt(double(dim));
    for (int i = 0; i < feature.size(); i++) {
        if (feature[i] > theta)
            feature[i] = theta;
        else if (feature[i] < -theta)
            feature[i] = -theta;
    }

    norm = sqrt(inner_product(feature.begin(), feature.end(), feature.begin(), 0.0));
    for (int i = 0; i < feature.size(); i++)
        feature[i] /= norm;
}