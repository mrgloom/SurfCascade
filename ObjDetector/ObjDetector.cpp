#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
#include <windows.h>
#include <vector>
#include <array>
#include <fstream>

using std::string;
using std::array;

int get_filepaths(string folder, string wildcard, vector<string>& filepaths);
void resize_patches(Size size1, Size size2, vector<vector<Rect>>& patches);

int main(int argc, char *argv[])
{
    //if (argc != 2)
    //{
    //    cout << "Usage: ObjDetector [--train|--detect]" << endl;
    //    return 0;
    //}
    CascadeClassifier cascade_classifier;

    /************************************************************************/
    /*                             Train Mode                               */
    /************************************************************************/
    //if (strcmp(argv[1], "--train") == 0 || strcmp(argv[1], "-t") == 0)
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
            dense_surf_feature_extractor.IntegralImage(filepaths[i], sums);
            dense_surf_feature_extractor.ExtractFeatures(sums, patches, features_img);
            features_all.push_back(features_img);
        }

        cout << "Training cascade classifier..." << endl;
        /* train cascade classifier */
        //CascadeClassifier cascade_classifier;
        cascade_classifier.Train(features_all, labels);
        cascade_classifier.Print();

        cout << "Done." << endl;
    }

    /************************************************************************/
    /*                             Detect Mode                              */
    /************************************************************************/
    //else if (strcmp(argv[1], "--detect") == 0 || strcmp(argv[1], "-d") == 0)
    {
        string filepath = "D:/Downloads/emily.jpg";

        /* extract patches */
        DenseSURFFeatureExtractor dense_surf_feature_extractor;
        vector<Rect> all_patches;

        cout << "Extracting patches..." << endl;
        Rect win(0, 0, 19, 19); //TODO
        dense_surf_feature_extractor.ExtractPatches(win, all_patches);

        /* get fitted patches */
        //CascadeClassifier cascade_classifier; // TODO: read trained classifier
        vector<vector<int>> patch_indexes;
        vector<vector<Rect>> patches;

        cout << "Getting fitted patches..." << endl;
        cascade_classifier.GetFittedPatchIndexes(patch_indexes);
        for (int i = 0; i < patch_indexes.size(); i++)
        {
            vector<Rect> patches_perstage;
            for (int j = 0; j < patch_indexes[0].size(); j++)
                patches_perstage.push_back(all_patches[patch_indexes[i][j]]);
            patches.push_back(patches_perstage);
        }

        /* calculate integral image */
        Mat sums[DenseSURFFeatureExtractor::n_bins];

        cout << "Calculating integral image..." << endl;
        dense_surf_feature_extractor.IntegralImage(filepath, sums);
        Size img(sums[0].cols - 1, sums[0].rows - 1);

        /* scan with varying windows */
        cout << "Scanning with varying windows..." << endl;
        for (Rect win(0, 0, 10, 10); win.width <= img.width && win.height <= img.height; win.width = int(win.width * 1.1), win.height = int(win.height * 1.1))
        {
            for (win.y = 0; win.y + win.height <= img.height; win.y += 2)
                for (win.x = 0; win.x + win.width <= img.width; win.x += 2)
                {
                    resize_patches(Size(19, 19), win.size(), patches);

                    vector<vector<vector<double>>> features_win;

                    for (int i = 0; i < patches.size(); i++)
                    {
                        vector<vector<double>> features_win_perstage;
                        dense_surf_feature_extractor.ExtractFeatures(sums, patches[i], features_win_perstage);
                        features_win.push_back(features_win_perstage);
                    }
                    if (cascade_classifier.Predict2(features_win))
                        cout << win << endl;
                }
            cout << win << endl;
        }
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

void resize_patches(Size size1, Size size2, vector<vector<Rect>>& patches)
{
    double scale = (double)size2.width / size1.width; // both square

    for (int i = 0; i < patches.size(); i++)
    {
        for (int j = 0; j < patches[0].size(); j++)
        {
            double ratio = (double)patches[i][j].width / patches[i][j].height;
            patches[i][j].x = (int)(patches[i][j].x * scale);
            patches[i][j].y = (int)(patches[i][j].y * scale);
            patches[i][j].height = (int)(patches[i][j].height * scale);
            patches[i][j].width = (int)(patches[i][j].height * ratio);
        }
    }
}