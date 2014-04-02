#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
#include <vector>
#include <string>
#include <array>
#include <numeric>
#include <xmmintrin.h>

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
using std::flush;

const Size DenseSURFFeatureExtractor::shapes[3] = { Size(2, 2), Size(1, 4), Size(4, 1) };

DenseSURFFeatureExtractor::~DenseSURFFeatureExtractor()
{
}

void DenseSURFFeatureExtractor::LoadFileList(string filename, string prefix_path, bool set_size)
{
    ifstream filestream;
    string imgname;

    this->prefix_path = prefix_path;
    filestream.open(prefix_path + filename);

    imgnames.clear();
    while (getline(filestream, imgname))
        imgnames.push_back(imgname);

    filestream.close();

    if (set_size)
    {
        Mat img;
        img = imread(prefix_path + imgnames[0], cv::IMREAD_GRAYSCALE);
        size = Size(img.cols, img.rows);
    }
}

void DenseSURFFeatureExtractor::ExtractPatches(vector<Rect>& patches)
{
    for (int j = 0; j < sizeof(shapes) / sizeof(shapes[0]); j++)
    {
        for (int cell_edge = min_cell_edge; cell_edge <= size.width / 2; cell_edge++)
        {
            Rect win(0, 0, shapes[j].width * cell_edge, shapes[j].height * cell_edge);

            for (win.y = 0; win.y + win.height <= size.height; win.y += step)
            for (win.x = 0; win.x + win.width <= size.width; win.x += step) {
                patches.push_back(win);
            }
        }
    }
}

void DenseSURFFeatureExtractor::IntegralImage(Mat img)
{
    Mat img_padded;
    Mat img_filtered(img.rows, img.cols, CV_8UC1);
    vector<Mat> sumvec(n_bins);

    sumtab = new F256Dat*[img.rows + 1];
    for (int i = 0; i < img.rows + 1; i++)
        sumtab[i] = new F256Dat[img.cols + 1];

    /* calculate integral image */
    copyMakeBorder(img, img_padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    for (int bin = 0; bin < n_bins; bin++) {
        T2bFilter(img_padded, img_filtered, bin);
        sumvec[bin].create(img.rows + 1, img.cols + 1, CV_32FC1);
        integral(img_filtered, sumvec[bin], CV_32FC1);
    }

    //__declspec(align(16))
    Mat sum;
    merge(sumvec, sum);

    for (int i = 0; i < img.rows + 1; i++)
    {
        for (int j = 0; j < img.cols + 1; j++)
        {
            sumtab[i][j].xmm_f1 = _mm_loadu_ps((float*)sum.ptr<float>(i) + j * 8);
            sumtab[i][j].xmm_f2 = _mm_loadu_ps((float*)sum.ptr<float>(i) + j * 8 + 4);
        }
    }
}

void DenseSURFFeatureExtractor::ExtractFeatures(const vector<Rect>& patches, vector<vector<float>>& features_win)
{
    /* compute features */
    for (int i = 0; i < patches.size(); i++)
        CalcFeature(patches[i], features_win[i]);
}

void DenseSURFFeatureExtractor::ExtractFeatures(const vector<vector<Rect>>& patches, vector<vector<vector<float>>>& features_win)
{
    for (int i = 0; i < patches.size(); i++)
        for (int j = 0; j < patches[i].size(); j++)
            CalcFeature(patches[i][j], features_win[i][j]);
}

bool DenseSURFFeatureExtractor::ExtractNextImageFeatures(const vector<Rect>& patches, vector<vector<float>>& features_img)
{
    static int i = 0;

    if (i < imgnames.size())
    {
        Mat img = imread(prefix_path + imgnames[i], cv::IMREAD_GRAYSCALE);
        assert(img.cols == size.width && img.rows == size.height);

        IntegralImage(img);
        ExtractFeatures(patches, features_img);

        i++;
        return true;
    }
    else
        return false;
}

bool DenseSURFFeatureExtractor::FillNegSamples(const vector<Rect>& patches, vector<vector<vector<float>>>& features_all, int n_total, CascadeClassifier& cascade_classifier, bool first)
{
    static int idx = 0;
    vector<Rect> new_patches(patches);
    vector<vector<float>> features_img(new_patches.size(), vector<float>(dim));

    bool done = false;

#pragma omp for schedule(dynamic)
    for (int i = idx; i < imgnames.size(); i++)
    {
        if (!done)
        {
            Mat img = imread(prefix_path + imgnames[i], cv::IMREAD_GRAYSCALE);
            LOG_DEBUG("\tReading image: " << imgnames[i] << ", features_all.size() = " << features_all.size());

            IntegralImage(img);

            Rect win(0, 0, size.width, size.height);
            for (win.y = 0; win.y + win.height <= img.size().height; win.y += win.height)
            {
                for (win.x = 0; win.x + win.width <= img.size().width; win.x += win.width)
                {
                    ProjectPatches(win, patches, new_patches);
                    ExtractFeatures(new_patches, features_img);

                    if (first == true || cascade_classifier.Predict(features_img) == true) // if false positive
                    {
#pragma omp critical
                        if (features_all.size() < n_total)
                        {
                            features_all.push_back(features_img);
                            LOG_INFO_NN("\r\tFilled: " << features_all.size() - n_total / 2 << '/' << n_total / 2 << flush);
                        }
                    }

                    if (features_all.size() == n_total)
                    {
                        done = true;
                        idx = i;
                    }
                }
            }

            for (int j = 0; j < img.rows + 1; j++)
                delete[] sumtab[j];
            delete[] sumtab;
        }
    }

    if (!done)
        LOG_WARNING("\tRunning out of negative samples.");
    return done;
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
            img_filtered.at<uchar>(y - 1, x - 1) = (abs(d) + d * (bin % 2 ? 1 : -1)) / 2;
        }
    }
}

void DenseSURFFeatureExtractor::CalcFeature(const Rect& patch, vector<float>& feature)
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
    for (int i = 0; i < n_cells; i++) {
        _mm_storeu_ps(feature.data() + i * 8, _mm_sub_ps(
            _mm_add_ps(sumtab[rects[i].y][rects[i].x].xmm_f1, sumtab[rects[i].y + rects[i].height][rects[i].x + rects[i].width].xmm_f1),
            _mm_add_ps(sumtab[rects[i].y][rects[i].x + rects[i].width].xmm_f1, sumtab[rects[i].y + rects[i].height][rects[i].x].xmm_f1)));
        _mm_storeu_ps(feature.data() + i * 8 + 4, _mm_sub_ps(
            _mm_add_ps(sumtab[rects[i].y][rects[i].x].xmm_f2, sumtab[rects[i].y + rects[i].height][rects[i].x + rects[i].width].xmm_f2),
            _mm_add_ps(sumtab[rects[i].y][rects[i].x + rects[i].width].xmm_f2, sumtab[rects[i].y + rects[i].height][rects[i].x].xmm_f2)));
    }

    /* normalization */
    float norm;
    float norm_theta;
    float divisor;
    float sum = FLT_EPSILON;
    float* p0 = feature.data();
    float* p1 = feature.data() + dim;
    float* p = p0;

    for (p = p0; p != p1; p++) {
        sum += *p * *p;
    }
    norm = sqrt(sum);
    norm_theta = norm * theta;

    sum = FLT_EPSILON;
    for (p = p0; p != p1; p++) {
        if (*p > norm_theta)
            *p = norm_theta;
        else if (*p < -norm_theta)
            *p = -norm_theta;

        sum += *p * *p;
    }
    norm = sqrt(sum);

    divisor = 1 / norm;
    for (p = p0; p != p1; p++)
        *p *= divisor;
}

void DenseSURFFeatureExtractor::ProjectPatches(const Rect win2, const vector<vector<Rect>>& patches1, vector<vector<Rect>>& patches2)
{
    float scale = (float)win2.width / size.width; // both square

    for (int i = 0; i < patches1.size(); i++)
    {
        for (int j = 0; j < patches1[i].size(); j++)
        {
            patches2[i][j].x = (int)(patches1[i][j].x * scale) + win2.x;
            patches2[i][j].y = (int)(patches1[i][j].y * scale) + win2.y;

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

void DenseSURFFeatureExtractor::ProjectPatches(const Rect win2, const vector<Rect>& patches1, vector<Rect>& patches2)
{
    float scale = (float)win2.width / size.width; // both square

    for (int i = 0; i < patches1.size(); i++)
    {
        patches2[i].x = (int)(patches1[i].x * scale) + win2.x;
        patches2[i].y = (int)(patches1[i].y * scale) + win2.y;

        if (patches1[i].width >= patches1[i].height)
        {
            int ratio = patches1[i].width / patches1[i].height;
            patches2[i].height = (int)(patches1[i].height * scale);
            patches2[i].width = patches2[i].height * ratio;
        }
        else
        {
            int ratio = patches1[i].height / patches1[i].width;
            patches2[i].width = (int)(patches1[i].width * scale);
            patches2[i].height = patches2[i].width * ratio;
        }
    }
}