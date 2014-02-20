#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
#include <windows.h>
#include <vector>
#include <array>
#include <fstream>

using std::string;
using std::array;

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

int main(int argc, char *argv[])
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
    Mat img = cv::imread(filepaths[0], cv::IMREAD_GRAYSCALE);
    Rect win(0, 0, img.cols, img.rows);
    dense_surf_feature_extractor.ExtractPatches(win, patches);

    /* extract features */
    vector<vector<vector<double>>> features_all;
    Mat sums[DenseSURFFeatureExtractor::n_bins];

    cout << "Extracting features..." << endl;
    for (int i = 0; i < filepaths.size(); i++)
    {
        vector<vector<double>> features_img;
        dense_surf_feature_extractor.IntegralImage(filepaths[i], sums);
        dense_surf_feature_extractor.ExtractFeatures(sums, patches, features_img);
        features_all.push_back(features_img);
    }

    cout << "Training cascade classifier..." << endl;
    /* train cascade classifier */
    CascadeClassifier cascade_classifier;
    cascade_classifier.Train(features_all, labels);
    cascade_classifier.Print();

    cout << "Done." << endl;




    //string filepath = "D:/Downloads/emily.jpg";

    ///* calculate integral image */
    //Mat sums[DenseSURFFeatureExtractor::n_bins];

    //DenseSURFFeatureExtractor dense_surf_feature_extractor;
    //dense_surf_feature_extractor.IntegralImage(filepath, sums);

    //Size img_size(sums[0].cols - 1, sums[0].rows - 1);

    ///* scan with variant windows */
    ////CascadeClassifier cascade_classifier;
    //// TODO: read trained classifier

    ////for (Rect win(0, 0, 10, 10); win.width <= img_size.width && win.height <= img_size.height; win.width = int(win.width * 1.1), win.height = int(win.height * 1.1))
    //for (Rect win(0, 0, 10, 10); win.width <= 11 && win.height <= 11; win.width = int(win.width * 1.1), win.height = int(win.height * 1.1))
    //{
    //    for (win.y = 0; win.y + win.height <= img_size.height; win.y += 2)
    //        for (win.x = 0; win.x + win.width <= img_size.width; win.x += 2)
    //        {
    //            vector<vector<double>> features_win;

    //            dense_surf_feature_extractor.ExtractFeatures(sums, win, features_win);
    //            //if (cascade_classifier.Predict(features_win))
    //            //cout << win << endl;
    //        }
    //    cout << win << endl;
    //}



    return 0;
}