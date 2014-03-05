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
using std::ifstream;

const Size DenseSURFFeatureExtractor::shapes[3] = { Size(2, 2), Size(1, 4), Size(4, 1) };

DenseSURFFeatureExtractor::~DenseSURFFeatureExtractor()
{
}

void DenseSURFFeatureExtractor::LoadFileList(string filename, string prefix_path)
{
    string imgname;
    Mat img;

    this->prefix_path = prefix_path;

    filestream.open(filename);
    filestream >> imgname;
    filestream.seekg(0, filestream.beg);

    img = imread(prefix_path + imgname, cv::IMREAD_GRAYSCALE);
    win = Rect(0, 0, img.cols, img.rows);
}

void DenseSURFFeatureExtractor::ExtractPatches(vector<Rect>& patches)
{
    Rect patch;

    for (int j = 0; j < sizeof(shapes) / sizeof(shapes[0]); j++) {
        Size shape = shapes[j];

        for (int cell_edge = min_cell_edge; cell_edge <= win.width / 2; cell_edge++) {
            for (int y = win.y; y + shape.height * cell_edge <= win.y + win.height; y += step)
            for (int x = win.x; x + shape.width * cell_edge <= win.x + win.width; x += step) {
                ConvertToPatch(x, y, shape, cell_edge, patch);
                patches.push_back(patch);
            }
        }
    }
}

void DenseSURFFeatureExtractor::IntegralImage(Mat img)
{
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

void DenseSURFFeatureExtractor::ExtractFeatures(const vector<Rect>& patches, vector<vector<double>>& features_win)
{
    /* compute features */
    vector<double> feature;

    for (int i = 0; i < patches.size(); i++)
    {
        CalcFeature(patches[i], feature);
        features_win.push_back(feature);
    }
}

void DenseSURFFeatureExtractor::ExtractFeatures(const vector<vector<Rect>>& patches, vector<vector<vector<double>>>& features_win)
{
    for (int i = 0; i < patches.size(); i++)
    {
        vector<vector<double>> features_win_perstage;

        for (int j = 0; j < patches[i].size(); j++)
        {
            vector<double> feature;
            CalcFeature(patches[i][j], feature);
            features_win_perstage.push_back(feature);
        }

        features_win.push_back(features_win_perstage);
    }
}

bool DenseSURFFeatureExtractor::ExtractNextImageFeatures(const vector<Rect>& patches, vector<vector<double>>& features_img)
{
    string imgname;

    features_img.clear();

    if (filestream >> imgname)
    {
        Mat img = imread(prefix_path + imgname, cv::IMREAD_GRAYSCALE);
        assert(img.cols == win.width && img.rows == win.height);

        IntegralImage(img);
        ExtractFeatures(patches, features_img);

        return true;
    }
    else
    {
        filestream.close();
        filestream.clear();
        return false;
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

void DenseSURFFeatureExtractor::ConvertToPatch(int x, int y, Size shape, int cell_edge, Rect& patch)
{
    patch.x = x;
    patch.y = y;
    patch.width = shape.width * cell_edge;
    patch.height = shape.height * cell_edge;
}

void DenseSURFFeatureExtractor::CalcFeature(Rect patch, vector<double>& feature)
{
    /* get separated blocks from patch */
    int cell_edge;
    if (patch.width == patch.height)
        cell_edge = patch.width / 2;
    else
        cell_edge = patch.width < patch.height ? patch.width : patch.height;

    Rect rects[n_cells];
    Size shape;
    shape.width = patch.width / cell_edge;
    shape.height = patch.height / cell_edge;

    for (int h = 0; h < shape.height; h++)
    for (int w = 0; w < shape.width; w++) {
        rects[h * shape.width + w] = Rect(patch.x + w * cell_edge, patch.y + h * cell_edge, cell_edge, cell_edge);
    }

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

void DenseSURFFeatureExtractor::ProjectPatches(Rect win1, Rect win2, const vector<vector<Rect>>& patches1, vector<vector<Rect>>& patches2)
{
    double scale = (double)win2.width / win1.width; // both square

    for (int i = 0; i < patches1.size(); i++)
    {
        for (int j = 0; j < patches1[i].size(); j++)
        {
            patches2[i][j].x = (int)(patches1[i][j].x * scale) + (win2.x - win1.x);
            patches2[i][j].y = (int)(patches1[i][j].y * scale) + (win2.y - win1.y);

            if (patches1[i][j].width >= patches1[i][j].height)
            {
                int ratio = patches1[i][j].width / patches1[i][j].height;
                patches2[i][j].height = (int)(patches1[i][j].height * scale);
                patches2[i][j].width = patches2[i][j].height * ratio;
            }
            else
            {
                int ratio = patches1[i][j].height / patches1[i][j].width;
                patches2[i][j].width = (int)(patches1[i][j].width * scale);
                patches2[i][j].height = patches2[i][j].width * ratio;
            }
        }
    }
}