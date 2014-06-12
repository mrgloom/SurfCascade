#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
#include "Model.h"
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <vector>
#include <array>
#include <fstream>

using std::ifstream;
using std::ofstream;
using std::ios;
using std::string;
using std::array;

int get_filepaths(string folder, string wildcard, vector<string>& filepaths);
void fast_nms(vector<Rect>& rects, vector<double>& scores, double overlap_th);

double get_wall_time(){
    LARGE_INTEGER time, freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time(){
    FILETIME a, b, c, d;
    if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
            (double)(d.dwLowDateTime |
            ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
    }
    else{
        //  Handle error
        return 0;
    }
}

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
        while (i < dense_surf_feature_extractor.imgnames.size())
            dense_surf_feature_extractor.ExtractNextImageFeatures(patches, features_all[i++]);

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
        bool show = 0;

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

        /* scan with varying windows */
        vector<Rect> wins;
        vector<double> scores;
        vector<vector<vector<float>>> features_win(patches.size());
        int step = win.width > 20 ? win.width / 20 : 1;

        for (int i = 0; i < features_win.size(); i++)
        {
            features_win[i].resize(patches[i].size(), vector<float>(dense_surf_feature_extractor.dim));
        }

        string prefix = "D:/FaceData/FDDB/";
        string filelist = "evaluations.txt";
        ifstream fs(prefix + filelist);
        vector<string> filepaths;

        while (getline(fs, filepath))
            filepaths.push_back(filepath);
        fs.close();

        ofstream of("C:/Users/Meng/Documents/MATLAB/surf.txt", ios::binary);

        double wall0 = get_wall_time();
        double cpu0 = get_cpu_time();

        for (int j = 0; j < filepaths.size(); j++)
        {
        cout << "Detecting image " << j + 1 << '/' << filepaths.size() << endl;
        //cout << "Calculating integral image..." << endl;
        Mat img = imread(prefix + filepaths[j] + ".jpg", cv::IMREAD_GRAYSCALE);
        dense_surf_feature_extractor.IntegralImage(img);

        /* prepare showing image */
        Mat img_rgb;
        if (show) {
            img_rgb.create(img.size(), CV_8UC3);
            cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2BGR);
        }

        int win_size_num = (int)min(log(img.size().width / (float)70) / log(1.1), log(img.size().height / (float)70) / log(1.1));

        //cout << "Scanning with varying windows..." << endl;
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
                    if (dense_surf_feature_extractor.sum(win) > win.area() * 6) {
                    dense_surf_feature_extractor.ProjectPatches(win, fitted_patches, patches);

                    double score;
                    int p;
                    for (p = 0; p < patches.size(); p++) {
                        for (int q = 0; q < patches[p].size(); q++)
                            dense_surf_feature_extractor.CalcFeature(patches[p][q], features_win[p][q]);

                        if ((score = cascade_classifier.stage_classifiers[p]->Predict2(features_win[p])) < cascade_classifier.stage_classifiers[p]->theta)
                            break;
                    }

                    score = (score + p + 1) / cascade_classifier.stage_classifiers.size();

                    if (p == cascade_classifier.stage_classifiers.size())
                    {
                        #pragma omp critical
                        {
                            wins.push_back(win);
                            scores.push_back(score);
                            //if (show)
                            //    rectangle(img_rgb, win, cv::Scalar(255, 0, 0), 1);
                        }
                    }

                    multi = (score < 0.5) ? 2 : 1;
                    }
                    else
                        multi = 2;
                }
            }
        }
        //cout << "Over." << endl;

        //fast_nms(wins, scores, 0.7);
        vector<int> weights(wins.size(), 0);
        groupRectangles(wins, weights, scores, 2, 0.2);
        //groupRectangles(wins, 2, 0.2);

        of << filepaths[j] << '\n';
        of << wins.size() << '\n';
        for (int k = 0; k < wins.size(); k++)
            of << wins[k].x << ' ' << wins[k].y << ' ' << wins[k].width << ' ' << wins[k].height << ' ' << scores[k] << '\n';

        if (show) {
            for (int k = 0; k < wins.size(); k++)
                rectangle(img_rgb, wins[k], cv::Scalar(0, 255, 0), 2);

            cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
            cv::imshow("Result", img_rgb);
            cv::waitKey(0);
        }

        wins.clear();
        scores.clear();
        }

        double wall1 = get_wall_time();
        double cpu1 = get_cpu_time();

        cout << "Wall Time = " << wall1 - wall0 << endl;
        cout << "CPU Time  = " << cpu1 - cpu0 << endl;

        of.close();
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
    vector<double> picked_scores;
    for (int i = 0; i < counter; ++i) {
        picked.push_back(rects[pick[i]]);
        picked_scores.push_back(scores[pick[i]]);
    }

    rects = picked;
    scores = picked_scores;

    free(pmem);
}
