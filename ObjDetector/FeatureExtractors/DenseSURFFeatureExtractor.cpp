#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
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
    uchar *grad = new uchar[img.rows * img.cols * n_bins];
    vector<Mat> sumvec(n_bins);

    /* calculate integral image */
    T2bFilter(img, grad);

    for (int bin = 0; bin < n_bins; bin++) {
        Mat grad1(img.rows, img.cols, CV_8UC1, grad + bin * img.rows * img.cols);
        integral(grad1, sumvec[bin], CV_32FC1);
    }

    //__declspec(align(16))

    merge(sumvec, summat);

    sumtab = new F256Dat*[img.rows + 1];
    for (int i = 0; i < img.rows + 1; i++)
        sumtab[i] = (F256Dat *)summat.data + i * (img.cols + 1);

    delete grad;
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

    for (int i = idx; i < imgnames.size(); i++)
    {
        Mat img = imread(prefix_path + imgnames[i], cv::IMREAD_GRAYSCALE);
        LOG_DEBUG("\tReading image: " << imgnames[i] << ", features_all.size() = " << features_all.size());

        if (!img.data || img.cols < size.width || img.rows < size.height)
            continue;

        IntegralImage(img);

        Rect win(0, 0, size.width, size.height);
        #pragma omp parallel for firstprivate(new_patches, features_img)
        for (int j = 0; j < (img.size().height - win.height) / win.height; j++)
        {
            if (!done)
            {
                win.y = j * win.height;
                for (win.x = 0; win.x + win.width <= img.size().width; win.x += win.width)
                {
                    if (!done)
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
                                if (features_all.size() == n_total) {
                                    done = true;
                                    idx = i + 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        delete[] sumtab;

        if (done)
            break;
    }

    if (!done)
        LOG_WARNING("\tRunning out of negative samples.");
    return done;
}

#define _mm_srli_epi8( _A, _Imm ) _mm_and_si128( _mm_set1_epi8((char)(0xFF >> _Imm)), _mm_srli_epi32( _A, _Imm ) )

void DenseSURFFeatureExtractor::T2bFilter(const Mat& img, uchar *grad)
{
    int w = img.cols;
    int h = img.rows;
    int sz = w * h;

    int d;
    uchar *I;
    uchar *Ip;
    uchar *In;
    uchar *G1;
    uchar *G2;
    __m128i *_Ip;
    __m128i *_In;
    __m128i *_G1;
    __m128i *_G2;

    int x;
    int w_sse;

    for (int y = 0; y < h; y++) {
        I = (uchar *)img.ptr(y);

        /* |dx| - dx, |dx| + dx */
        Ip = I;
        In = I + 1;
        G1 = grad + y * w;
        G2 = G1 + sz;

        d = *In++ - *Ip;
        *G1++ = (abs(d) - d) / 2;
        *G2++ = (abs(d) + d) / 2;

        _Ip = (__m128i *)Ip; _In = (__m128i *)In; _G1 = (__m128i *)G1; _G2 = (__m128i *)G2;

        w_sse = (((w - 2) >> 4) << 4) + 1;
        for (x = 1; x < w_sse; x += 16, _Ip++, _In++)
        {
            _mm_storeu_si128(_G1++, _mm_subs_epu8(_mm_loadu_si128(_Ip), _mm_loadu_si128(_In)));
            _mm_storeu_si128(_G2++, _mm_subs_epu8(_mm_loadu_si128(_In), _mm_loadu_si128(_Ip)));
        }

        Ip = (uchar *)_Ip; In = (uchar *)_In; G1 = (uchar *)_G1; G2 = (uchar *)_G2;

        for (; x < w - 1; x++) {
            d = *In++ - *Ip++;
            *G1++ = (abs(d) - d) / 2;
            *G2++ = (abs(d) + d) / 2;
        }
        d = *--In - *Ip;
        *G1 = (abs(d) - d) / 2;
        *G2 = (abs(d) + d) / 2;

        /* |dy| - dy, |dy| + dy */
        Ip = I - w;
        In = I + w;
        G1 = grad + 2 * sz + y * w;
        G2 = G1 + sz;

        if (y == 0) Ip += w;
        else if (y == h - 1) In -= w;

        _Ip = (__m128i *)Ip; _In = (__m128i *)In; _G1 = (__m128i *)G1; _G2 = (__m128i *)G2;

        w_sse = (w >> 4) << 4;
        for (x = 0; x < w_sse; x += 16, _Ip++, _In++)
        {
            _mm_storeu_si128(_G1++, _mm_subs_epu8(_mm_loadu_si128(_Ip), _mm_loadu_si128(_In)));
            _mm_storeu_si128(_G2++, _mm_subs_epu8(_mm_loadu_si128(_In), _mm_loadu_si128(_Ip)));
        }

        Ip = (uchar *)_Ip; In = (uchar *)_In; G1 = (uchar *)_G1; G2 = (uchar *)_G2;

        for (; x < w; x++)
        {
            d = *In++ - *Ip++;
            *G1++ = (abs(d) - d) / 2;
            *G2++ = (abs(d) + d) / 2;
        }

        /* |du| - du, |du| + du */
        Ip = I - w;
        In = I + w + 1;
        G1 = grad + 4 * sz + y * w;
        G2 = G1 + sz;

        if (y == 0) Ip += w;
        else if (y == h - 1) In -= w;

        d = *In++ - *Ip;
        *G1++ = (abs(d) - d) / 2;
        *G2++ = (abs(d) + d) / 2;

        _Ip = (__m128i *)Ip; _In = (__m128i *)In; _G1 = (__m128i *)G1; _G2 = (__m128i *)G2;

        w_sse = (((w - 2) >> 4) << 4) + 1;
        for (x = 1; x < w_sse; x += 16, _Ip++, _In++)
        {
            _mm_storeu_si128(_G1++, _mm_subs_epu8(_mm_loadu_si128(_Ip), _mm_loadu_si128(_In)));
            _mm_storeu_si128(_G2++, _mm_subs_epu8(_mm_loadu_si128(_In), _mm_loadu_si128(_Ip)));
        }

        Ip = (uchar *)_Ip; In = (uchar *)_In; G1 = (uchar *)_G1; G2 = (uchar *)_G2;

        for (; x < w - 1; x++) {
            d = *In++ - *Ip++;
            *G1++ = (abs(d) - d) / 2;
            *G2++ = (abs(d) + d) / 2;
        }
        d = *--In - *Ip;
        *G1 = (abs(d) - d) / 2;
        *G2 = (abs(d) + d) / 2;

        /* |dv| - dv, |dv| + dv */
        Ip = I + w;
        In = I - w + 1;
        G1 = grad + 6 * sz + y * w;
        G2 = G1 + sz;

        if (y == 0) In += w;
        else if (y == h - 1) Ip -= w;

        d = *In++ - *Ip;
        *G1++ = (abs(d) - d) / 2;
        *G2++ = (abs(d) + d) / 2;

        _Ip = (__m128i *)Ip; _In = (__m128i *)In; _G1 = (__m128i *)G1; _G2 = (__m128i *)G2;

        w_sse = (((w - 2) >> 4) << 4) + 1;
        for (x = 1; x < w_sse; x += 16, _Ip++, _In++)
        {
            _mm_storeu_si128(_G1++, _mm_subs_epu8(_mm_loadu_si128(_Ip), _mm_loadu_si128(_In)));
            _mm_storeu_si128(_G2++, _mm_subs_epu8(_mm_loadu_si128(_In), _mm_loadu_si128(_Ip)));
        }

        Ip = (uchar *)_Ip; In = (uchar *)_In; G1 = (uchar *)_G1; G2 = (uchar *)_G2;

        for (; x < w - 1; x++) {
            d = *In++ - *Ip++;
            *G1++ = (abs(d) - d) / 2;
            *G2++ = (abs(d) + d) / 2;
        }
        d = *--In - *Ip;
        *G1 = (abs(d) - d) / 2;
        *G2 = (abs(d) + d) / 2;
    }
}

void GetRectsFromPatch(const Rect& patch, Rect rects[])
{
    /* get separated blocks from patch */
    int cell_edge;
    if (patch.width == patch.height)
        cell_edge = patch.width / 2;
    else
        cell_edge = patch.width < patch.height ? patch.width : patch.height;

    Size shape;
    shape.width = patch.width / cell_edge;
    shape.height = patch.height / cell_edge;

    for (int h = 0; h < shape.height; h++)
    for (int w = 0; w < shape.width; w++) {
        rects[h * shape.width + w] = Rect(patch.x + w * cell_edge, patch.y + h * cell_edge, cell_edge, cell_edge);
    }
}

void DenseSURFFeatureExtractor::CalcFeature(const Rect& patch, vector<float>& feature)
{
    Rect rects[n_cells];
    GetRectsFromPatch(patch, rects);

    /* calculate feature value using integral image*/
    _mm_storeu_ps(feature.data() + 0 * 8, _mm_sub_ps(
        _mm_add_ps(sumtab[rects[0].y][rects[0].x].xmm_f1, sumtab[rects[0].y + rects[0].height][rects[0].x + rects[0].width].xmm_f1),
        _mm_add_ps(sumtab[rects[0].y][rects[0].x + rects[0].width].xmm_f1, sumtab[rects[0].y + rects[0].height][rects[0].x].xmm_f1)));
    _mm_storeu_ps(feature.data() + 0 * 8 + 4, _mm_sub_ps(
        _mm_add_ps(sumtab[rects[0].y][rects[0].x].xmm_f2, sumtab[rects[0].y + rects[0].height][rects[0].x + rects[0].width].xmm_f2),
        _mm_add_ps(sumtab[rects[0].y][rects[0].x + rects[0].width].xmm_f2, sumtab[rects[0].y + rects[0].height][rects[0].x].xmm_f2)));

    _mm_storeu_ps(feature.data() + 1 * 8, _mm_sub_ps(
        _mm_add_ps(sumtab[rects[1].y][rects[1].x].xmm_f1, sumtab[rects[1].y + rects[1].height][rects[1].x + rects[1].width].xmm_f1),
        _mm_add_ps(sumtab[rects[1].y][rects[1].x + rects[1].width].xmm_f1, sumtab[rects[1].y + rects[1].height][rects[1].x].xmm_f1)));
    _mm_storeu_ps(feature.data() + 1 * 8 + 4, _mm_sub_ps(
        _mm_add_ps(sumtab[rects[1].y][rects[1].x].xmm_f2, sumtab[rects[1].y + rects[1].height][rects[1].x + rects[1].width].xmm_f2),
        _mm_add_ps(sumtab[rects[1].y][rects[1].x + rects[1].width].xmm_f2, sumtab[rects[1].y + rects[1].height][rects[1].x].xmm_f2)));

    _mm_storeu_ps(feature.data() + 2 * 8, _mm_sub_ps(
        _mm_add_ps(sumtab[rects[2].y][rects[2].x].xmm_f1, sumtab[rects[2].y + rects[2].height][rects[2].x + rects[2].width].xmm_f1),
        _mm_add_ps(sumtab[rects[2].y][rects[2].x + rects[2].width].xmm_f1, sumtab[rects[2].y + rects[2].height][rects[2].x].xmm_f1)));
    _mm_storeu_ps(feature.data() + 2 * 8 + 4, _mm_sub_ps(
        _mm_add_ps(sumtab[rects[2].y][rects[2].x].xmm_f2, sumtab[rects[2].y + rects[2].height][rects[2].x + rects[2].width].xmm_f2),
        _mm_add_ps(sumtab[rects[2].y][rects[2].x + rects[2].width].xmm_f2, sumtab[rects[2].y + rects[2].height][rects[2].x].xmm_f2)));

    _mm_storeu_ps(feature.data() + 3 * 8, _mm_sub_ps(
        _mm_add_ps(sumtab[rects[3].y][rects[3].x].xmm_f1, sumtab[rects[3].y + rects[3].height][rects[3].x + rects[3].width].xmm_f1),
        _mm_add_ps(sumtab[rects[3].y][rects[3].x + rects[3].width].xmm_f1, sumtab[rects[3].y + rects[3].height][rects[3].x].xmm_f1)));
    _mm_storeu_ps(feature.data() + 3 * 8 + 4, _mm_sub_ps(
        _mm_add_ps(sumtab[rects[3].y][rects[3].x].xmm_f2, sumtab[rects[3].y + rects[3].height][rects[3].x + rects[3].width].xmm_f2),
        _mm_add_ps(sumtab[rects[3].y][rects[3].x + rects[3].width].xmm_f2, sumtab[rects[3].y + rects[3].height][rects[3].x].xmm_f2)));

    Normalize(feature);
}

void DenseSURFFeatureExtractor::Normalize(vector<float>& feature)
{
    /* normalization */
    float* p0 = feature.data();
    float* p1 = feature.data() + dim;
    float* p = p0;

    __m128 _s, _t, _t2;

    _s = _mm_set_ps(FLT_EPSILON, 0, 0, 0);
    for (p = p0; p != p1; p+=4)
        _s = _mm_hadd_ps(_s, _mm_mul_ps(_mm_loadu_ps(p), _mm_loadu_ps(p)));
    _s = _mm_hadd_ps(_s, _s);
    _s = _mm_hadd_ps(_s, _s);

    _t = _mm_mul_ps(_mm_sqrt_ps(_s), _mm_set_ps1(theta));
    _t2 = _mm_xor_ps(_t, _mm_set1_ps(-0.f));

    _s = _mm_set_ps(FLT_EPSILON, 0, 0, 0);
    for (p = p0; p != p1; p+=4) {
        _mm_storeu_ps(p, _mm_min_ps(_mm_loadu_ps(p), _t));
        _mm_storeu_ps(p, _mm_max_ps(_mm_loadu_ps(p), _t2));

        _s = _mm_hadd_ps(_s, _mm_mul_ps(_mm_loadu_ps(p), _mm_loadu_ps(p)));
    }
    _s = _mm_hadd_ps(_s, _s);
    _s = _mm_hadd_ps(_s, _s);

    _t = _mm_div_ps(_mm_set_ps1(1), _mm_sqrt_ps(_s));
    for (p = p0; p != p1; p += 4)
        _mm_storeu_ps(p, _mm_mul_ps(_mm_loadu_ps(p), _t));
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
