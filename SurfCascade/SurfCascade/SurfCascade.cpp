#include "SurfCascade.h"
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>

using namespace std;
using namespace cv;

static const Size win_size = { 19, 19 };

int main(int argc, char *argv[])
{
    /* get file names */
    string filepath = "D:/facedata/train/face/";
    string files = filepath + string("*.pgm");
    WIN32_FIND_DATA ffd;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    vector<string> pos_files;

    hFind = FindFirstFile(files.c_str(), &ffd);
    if (hFind == INVALID_HANDLE_VALUE) return -1;
    do {
        pos_files.push_back(string(ffd.cFileName));
    } while (FindNextFile(hFind, &ffd) != 0);
    FindClose(hFind);

    /* read images */
    vector<Mat> pos_imgs;

    cout << "Reading files..." << endl;
    for (vector<string>::iterator it = pos_files.begin(); it != pos_files.end(); ++it) {
        pos_imgs.push_back(imread(filepath + *it, IMREAD_GRAYSCALE));
    }

    /* iterate images */
    Mat sums[n_bins];
    Mat img_padded;
    Mat img_filtered(win_size.height, win_size.width, CV_8UC1);

    cout << "Iterating images..." << endl;
    for (vector<Mat>::size_type i = 0; i != pos_imgs.size(); i++) {
        Mat img = pos_imgs[i];

        /* calculate integral image */
        copyMakeBorder(img, img_padded, 1, 1, 1, 1, BORDER_REPLICATE);

        for (int bin = 0; bin < n_bins; bin++) {
            t2b_filter(img_padded, img_filtered, bin);
            sums[bin].create(win_size.height + 1, win_size.width + 1, CV_32SC1);
            integral(img_filtered, sums[bin]);
        }

        /* compute features */
        Rect rects[4];
        vector<array<float, dim>> features;

        for (int j = 0; j < sizeof(shapes) / sizeof(shapes[0]); j++) {
            Size shape = shapes[j];

            for (int cell_edge = min_cell_edge; cell_edge <= win_size.width / 2; cell_edge++) {
                for (int y = 0; y + shape.height * cell_edge < win_size.height; y += step)
                for (int x = 0; x + shape.width * cell_edge < win_size.width; x += step) {
                    get_feature_rects(x, y, shape, cell_edge, rects);
                    array<float, dim> feature;
                    calc_feature_value(sums, rects, feature);
                    features.push_back(feature);
                }
            }
        }

        /* */
    }

    return 0;
}

void t2b_filter(const Mat& img_padded, Mat& img_filtered, int bin)
{
    int d;
    for (int y = 1; y < win_size.height + 1; y++) {
        for (int x = 1; x < win_size.width + 1; x++) {
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

void get_feature_rects(int x, int y, Size shape, int cell_edge, Rect rects[])
{
    for (int h = 0; h < shape.height; h++)
    for (int w = 0; w < shape.width; w++) {
        rects[h * shape.width + w] = Rect(x + w * cell_edge, y + h * cell_edge, cell_edge, cell_edge);
    }
}

void calc_feature_value(const Mat sums[], Rect rects[], array<float, dim>& feature)
{
    /* calculate feature value using integral image*/
    int s0, s1, s2, s3, s;
    for (int i = 0; i < n_cells; i++) {
        for (int j = 0; j < n_bins; j++) {
            Mat sum = sums[j];
            s0 = sum.at<int>(rects[i].x, rects[i].y);
            s1 = sum.at<int>(rects[i].x + rects[i].width, rects[i].y);
            s2 = sum.at<int>(rects[i].x, rects[i].y + rects[i].height);
            s3 = sum.at<int>(rects[i].x + rects[i].width, rects[i].y + rects[i].height);
            s = s3 - s2 - s1 + s0;
            feature[i * j] = (float)s;
        }
    }

    /* normalization */
    normalization(feature);
}

void normalization(array<float, dim>& feature) {
    float norm;
    norm = sqrt(inner_product(feature.begin(), feature.end(), feature.begin(), (float)0));
    for (array<float, dim>::size_type i = 0; i < feature.size(); i++)
        feature[i] /= norm;

    float theta = 2 / sqrt(float(dim));
    for (array<float, dim>::size_type i = 0; i < feature.size(); i++) {
        if (feature[i] > theta)
            feature[i] = theta;
        else if (feature[i] < -theta)
            feature[i] = -theta;
    }

    norm = sqrt(inner_product(feature.begin(), feature.end(), feature.begin(), (float)0));
    for (array<float, dim>::size_type i = 0; i < feature.size(); i++)
        feature[i] /= norm;
}