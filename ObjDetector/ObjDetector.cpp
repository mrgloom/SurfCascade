#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
#include "Model.h"
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <vector>
#include <array>
#include <fstream>

using std::ifstream;
using std::string;
using std::array;

int get_filepaths(string folder, string wildcard, vector<string>& filepaths);
void fast_nms(vector<Rect>& rects, vector<double>& scores, double overlap_th);

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Usage: ObjDetector [--train|--detect]" << endl;
        return 0;
    }

    Model model("model.cfg");

    /************************************************************************/
    /*                             Train Mode                               */
    /************************************************************************/
    if (strcmp(argv[1], "--train") == 0 || strcmp(argv[1], "-t") == 0)
    {
        string prefix_path = "D:/FaceData/Custom/";
        string pos_file("facepos.list");
        string neg_file("faceneg.list");

        /* extract patches */
        DenseSURFFeatureExtractor dense_surf_feature_extractor;
        vector<Rect> patches;

        cout << "Extracting patches..." << endl;
        dense_surf_feature_extractor.LoadFileList(pos_file, prefix_path, true);
        dense_surf_feature_extractor.ExtractPatches(patches);

        /* extract features in positive samples */
        vector<vector<vector<float>>> features_all(dense_surf_feature_extractor.imgnames.size(), vector<vector<float>>(patches.size(), vector<float>(dense_surf_feature_extractor.dim)));

        cout << "Extracting features in positive samples..." << endl;
        int i = 0;
        while (dense_surf_feature_extractor.ExtractNextImageFeatures(patches, features_all[i++]))
            ;

        vector<bool> labels(features_all.size(), true);

        /* train cascade classifier */
        cout << "Training cascade classifier..." << endl;
        CascadeClassifier cascade_classifier;
        cascade_classifier.Train(features_all, labels, dense_surf_feature_extractor, neg_file, patches);
        cascade_classifier.Print();

        /* saving model to model.cfg... */
        model.Save(cascade_classifier);

        cout << "Done." << endl;
    }

    /************************************************************************/
    /*                             Detect Mode                              */
    /************************************************************************/
    else if (strcmp(argv[1], "--detect") == 0 || strcmp(argv[1], "-d") == 0)
    {
        string filepath = "D:/FaceData/Custom/Detect/9.jpg";
        int length = 70;
        Rect win(0, 0, length, length);

        /* extract patches */
        DenseSURFFeatureExtractor dense_surf_feature_extractor;
        vector<Rect> dense_patches;

        cout << "Extracting patches..." << endl;
        dense_surf_feature_extractor.size = Size(40, 40); // TODO
        dense_surf_feature_extractor.ExtractPatches(dense_patches);

        /* get fitted patches */
        CascadeClassifier cascade_classifier;
        model.Load(cascade_classifier);

        vector<vector<int>> patch_indexes;
        vector<vector<Rect>> fitted_patches;

        cout << "Getting fitted patches..." << endl;
        cascade_classifier.GetFittedPatchIndexes(patch_indexes);
        for (int i = 0; i < patch_indexes.size(); i++)
        {
            vector<Rect> patches_perstage;
            for (int j = 0; j < patch_indexes[i].size(); j++)
                patches_perstage.push_back(dense_patches[patch_indexes[i][j]]);
            fitted_patches.push_back(patches_perstage);
        }

        /* calculate integral image */
        vector<vector<Rect>> patches(fitted_patches);

        cout << "Calculating integral image..." << endl;
        Mat img = imread(filepath, cv::IMREAD_GRAYSCALE);
        dense_surf_feature_extractor.IntegralImage(img);

        /* prepare showing image */
        Mat img_rgb(img.size(), CV_8UC3);
        cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2BGR);

        /* scan with varying windows */
        vector<Rect> wins;
        vector<double> scores;
        vector<vector<vector<float>>> features_win(patches.size());
        int step = win.width > 20 ? win.width / 20 : 1;

        for (int i = 0; i < features_win.size(); i++)
        {
            features_win[i].resize(patches[i].size(), vector<float>(dense_surf_feature_extractor.dim));
        }

        int win_size_num = (int)min(log(img.size().width / (float)70) / log(1.1), log(img.size().height / (float)70) / log(1.1));

        cout << "Scanning with varying windows..." << endl;
        #pragma omp parallel for firstprivate(patches, features_win)
        for (int i = 0; i <= win_size_num; i++)
        {
            int l = (int)(70 * pow(1.1, i));
            Rect win(0, 0, l, l);
            for (int y = 0; y <= img.size().height - win.height; y += step)
            {
                win.y = y;
                int multi = 1;
                for (win.x = 0; win.x <= img.size().width - win.width; win.x += multi * step)
                {
                    dense_surf_feature_extractor.ProjectPatches(win, fitted_patches, patches);
                    dense_surf_feature_extractor.ExtractFeatures(patches, features_win);

                    double score;
                    if (cascade_classifier.Predict2(features_win, score))
                    {
                        #pragma omp critical
                        {
                            wins.push_back(win);
                            scores.push_back(score);
                            rectangle(img_rgb, win, cv::Scalar(255, 0, 0), 1);
                        }
                    }

                    multi = (score < cascade_classifier.stage_classifiers.size() / 2) ? 2 : 1;
                }
            }
        }
        cout << "Over." << endl;

        //fast_nms(wins, scores, 0.7);
        groupRectangles(wins, 2, 0.2);
        for (int i = 0; i < wins.size(); i++)
            rectangle(img_rgb, wins[i], cv::Scalar(0, 255, 0), 2);

        cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
        cv::imshow("Result", img_rgb);
        cv::waitKey(0);
    }

    return 0;
}

int get_filepaths(string folder, string wildcard, vector<string>& filepaths)
{
    WIN32_FIND_DATA ffd;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    int i = 0;

    hFind = FindFirstFile((folder + wildcard).c_str(), &ffd);
    if (hFind == INVALID_HANDLE_VALUE) return i;
    do {
        filepaths.push_back(folder + string(ffd.cFileName));
        i++;
    } while (FindNextFile(hFind, &ffd) != 0);
    FindClose(hFind);

    return i;
}

static void
sort_idx(const vector<double>& scores, int* idxes) {
    /* sort indexes descending by scores */
    int i, j;
    for (i = 0; i < scores.size(); ++i) {
        for (j = i + 1; j < scores.size(); ++j) {
            int ti = idxes[i], tj = idxes[j];
            if (scores[tj] < scores[ti]) {
                idxes[i] = tj;
                idxes[j] = ti;
            }
        }
    }
}

static int
sort_stable(int *arr, int n) {
    /* stable move all -1 to the end */
    int i = 0, j = 0;

    while (i < n) {
        if (arr[i] == -1) {
            if (j < i + 1)
                j = i + 1;
            while (j < n) {
                if (arr[j] == -1) ++j;
                else {
                    arr[i] = arr[j];
                    arr[j] = -1;
                    j++;
                    break;
                }
            }
            if (j == n) return i;
        }
        ++i;
    }
    return i;
}

#define fast_max(x,y) (x - ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))))
#define fast_min(x,y) (y + ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))))

void
fast_nms(vector<Rect>& rects, vector<double>& scores, double overlap_th) {
    size_t num = rects.size();

    void *pmem = malloc(sizeof(int)* (num + num) + sizeof(float)* num);
    int *idxes = (int *)pmem;
    int *pick = idxes + num;
    float *invareas = (float *)(pick + num);
    int idx_count = (int)num;
    int counter = 0, last_idx;
    int x0, y0, x1, y1;
    int tx0, ty0, tx1, ty1;

    for (int i = 0; i < num; ++i) idxes[i] = i;
    sort_idx(scores, idxes);

    for (int i = 0; i < num; ++i) {
        int ti = idxes[i];
        Rect r = rects[idxes[i]];
        invareas[ti] = 1.0f / ((r.width + 1) * (r.height + 1));
    }

    while (idx_count > 0) {
        int tmp_idx = idx_count - 1;
        last_idx = idxes[tmp_idx];
        pick[counter++] = last_idx;

        x0 = rects[last_idx].x;
        y0 = rects[last_idx].y;
        x1 = rects[last_idx].x + rects[last_idx].width;
        y1 = rects[last_idx].y + rects[last_idx].height;

        idxes[tmp_idx] = -1;

        for (int i = tmp_idx - 1; i != -1; i--) {

            Rect r = rects[idxes[i]];
            tx0 = fast_max(x0, r.x);
            ty0 = fast_max(y0, r.y);
            tx1 = fast_min(x1, r.x + r.width);
            ty1 = fast_min(y1, r.y + r.height);

            tx0 = tx1 - tx0 + 1;
            ty0 = ty1 - ty0 + 1;
            if (tx0 > 0 && ty0 > 0) {
                if (tx0 * ty0 * invareas[idxes[i]] > overlap_th) {
                    idxes[i] = -1;
                }
            }
        }
        idx_count = sort_stable(idxes, idx_count);
    }

    /* just give the selected rects' indexes, modification needed for real use */
    vector<Rect> picked;
    for (int i = 0; i < counter; ++i) {
        picked.push_back(rects[pick[i]]);
    }

    rects = picked;

    free(pmem);
}
