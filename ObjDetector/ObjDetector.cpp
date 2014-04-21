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
        string neg_file("neg.list");

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
        string filepath = "D:/FaceData/Custom/Detect/8.jpg";
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
                            rectangle(img_rgb, win, cv::Scalar(255, 0, 0), 1);
                        }
                    }

                    multi = (score < cascade_classifier.stage_classifiers.size() / 2) ? 2 : 1;
                }
            }
        }
        cout << "Over." << endl;

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