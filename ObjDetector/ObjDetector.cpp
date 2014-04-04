#include "FeatureExtractors/DenseSURFFeatureExtractor.h"
#include "LearningAlgorithms/CascadeClassifier/CascadeClassifier.h"
#include "LOG.h"
#include "Model.h"
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <vector>
#include <array>
#include <fstream>
#include <ctime>

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

        /* scan with varying windows */
        vector<Rect> wins;
        vector<vector<vector<float>>> features_win(patches.size());
        int step = win.width > 20 ? win.width / 20 : 1;

        for (int i = 0; i < features_win.size(); i++)
        {
            features_win[i].resize(patches[i].size(), vector<float>(dense_surf_feature_extractor.dim));
        }

        string prefix = "D:/FaceData/Custom/640x480/";
        string filelist = "filelist";
        ifstream fs(prefix + filelist);
        vector<string> filepaths;

        while (getline(fs, filepath))
            filepaths.push_back(prefix + filepath);

        clock_t start = clock();
        for (int i = 0; i < filepaths.size(); i++)
        {
        cout << "Calculating integral image..." << i << endl;
        Mat img = imread(filepaths[i], cv::IMREAD_GRAYSCALE);
        dense_surf_feature_extractor.IntegralImage(img);

        //cout << "Scanning with varying windows..." << endl;
        //for (Rect win(0, 0, 70, 70); win.width <= img.size().width && win.height <= img.size().height; win.width = int(win.width * 1.1), win.height = int(win.height * 1.1))
        //{
        //    for (int y = 0; y <= img.size().height - win.height; y += step)
        //    {
        //        win.y = y;
        //        int multi = 1;
        //        for (win.x = 0; win.x <= img.size().width - win.width; win.x += multi * step)
        //        {
        //            dense_surf_feature_extractor.ProjectPatches(win, fitted_patches, patches);
        //            dense_surf_feature_extractor.ExtractFeatures(patches, features_win);
        //        }
        //    }
        //}
        //cout << "Over." << endl;
        }
        cout << "Execution time: " << (clock() - start) / (CLOCKS_PER_SEC / 1000) << " milliseconds." << endl;
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