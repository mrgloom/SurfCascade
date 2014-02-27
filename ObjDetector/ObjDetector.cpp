#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
#include "Model.h"
#include <windows.h>
#include <vector>
#include <array>
#include <fstream>

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
        string pos_folder = "D:/facedata/train1/face/";
        string neg_folder = "D:/facedata/train1/non-face/";
        string wildcard = string("*.pgm");

        /* get file names and labels */
        vector<string> filepaths;
        vector<bool> labels;
        int n;

        n = get_filepaths(pos_folder, wildcard, filepaths);
        labels.assign(n, true);

        n = get_filepaths(neg_folder, wildcard, filepaths);
        vector<bool> neg_labels(n, false);
        labels.insert(labels.end(), neg_labels.begin(), neg_labels.end());

        /* extract patches */
        DenseSURFFeatureExtractor dense_surf_feature_extractor;
        vector<Rect> patches;

        cout << "Extracting patches..." << endl;
        Mat im0 = cv::imread(filepaths[0], cv::IMREAD_GRAYSCALE);
        Rect win(0, 0, im0.cols, im0.rows);
        dense_surf_feature_extractor.ExtractPatches(win, patches);

        /* extract features */
        vector<vector<vector<double>>> features_all;
        Mat sums[DenseSURFFeatureExtractor::n_bins];

        cout << "Extracting features..." << endl;
        for (int i = 0; i < filepaths.size(); i++)
        {
            vector<vector<double>> features_img;
            Mat img = imread(filepaths[i], cv::IMREAD_GRAYSCALE);
            dense_surf_feature_extractor.IntegralImage(img, sums);
            dense_surf_feature_extractor.ExtractFeatures(sums, patches, features_img);
            features_all.push_back(features_img);
        }

        /* train cascade classifier */
        cout << "Training cascade classifier..." << endl;
        CascadeClassifier cascade_classifier;
        cascade_classifier.Train(features_all, labels);
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
        string filepath = "D:/facedata/16-1.jpg";

        /* extract patches */
        DenseSURFFeatureExtractor dense_surf_feature_extractor;
        vector<Rect> all_patches;

        cout << "Extracting patches..." << endl;
        Rect win(0, 0, 19, 19); //TODO
        dense_surf_feature_extractor.ExtractPatches(win, all_patches);

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
                patches_perstage.push_back(all_patches[patch_indexes[i][j]]);
            fitted_patches.push_back(patches_perstage);
        }

        /* calculate integral image */
        Mat sums[DenseSURFFeatureExtractor::n_bins];
        vector<vector<Rect>> patches(fitted_patches);

        cout << "Calculating integral image..." << endl;
        Mat img = imread(filepath, cv::IMREAD_GRAYSCALE);
        Mat img_rgb(img.size(), CV_8UC3);
        cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2BGR);

        dense_surf_feature_extractor.IntegralImage(img, sums);
        Size imgsize(sums[0].cols - 1, sums[0].rows - 1);

        cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);

        /* scan with varying windows */
        cout << "Scanning with varying windows..." << endl;
        for (Rect win(0, 0, 19, 19); win.width <= imgsize.width && win.height <= imgsize.height; win.width = int(win.width * 1.1), win.height = int(win.height * 1.1))
        {
            for (win.y = 0; win.y + win.height <= imgsize.height; win.y += 19)
                for (win.x = 0; win.x + win.width <= imgsize.width; win.x += 2)
                {
                    dense_surf_feature_extractor.project_patches(Rect(0, 0, 19, 19), win, fitted_patches, patches);

                    vector<vector<vector<double>>> features_win;

                    for (int i = 0; i < patches.size(); i++)
                    {
                        vector<vector<double>> features_win_perstage;
                        dense_surf_feature_extractor.ExtractFeatures(sums, patches[i], features_win_perstage);
                        features_win.push_back(features_win_perstage);
                    }
                    if (cascade_classifier.Predict2(features_win))
                    {
                        cout << "Detected: " << win << endl;
                        rectangle(img_rgb, win, cv::Scalar(255, 0, 0), 1);
                    }

                    Mat img_tmp = img_rgb.clone();
                    rectangle(img_tmp, win, cv::Scalar(0, 255, 255), 1);
                    cv::imshow("Result", img_tmp);
                    cv::waitKey(100);
                }
        }
        cout << "Over.";

        cv::destroyWindow("Result");
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